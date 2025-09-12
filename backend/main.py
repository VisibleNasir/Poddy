import glob
import cv2
import ffmpegcv
import modal
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Depends ,HTTPException,status
from pydantic import BaseModel
import os
import json
import pathlib
import subprocess
import time
import uuid
import boto3
import numpy as np
import shutil
import tqdm
import whisperx
import pickle
from google import genai


class ProcessVideoRequest(BaseModel):
    s3_key: str


image = (modal.Image.from_registry(
    "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.12")
    .apt_install(["ffmpeg" , "libgl1-mesa-glx" , "wget" , "libcudnn8" , "libcudnn8-dev"])
    .pip_install_from_requirements("requirements.txt")
    .run_commands(["mkdir -p /user/share/fonts/truetype/custom","wget -O /user/share/fonts/truetype/custom/Anton-Regular.ttf  https://github.com/google/fonts/raw/main/ofl/anton/Anton-Regular.ttf","fc-cache -f -v"]).add_local_dir("asd","/asd",copy= True))

app = modal.App("ai-podcast-clipper" , image = image)


volume = modal.Volume.from_name(
    "ai-podcast-clipper-modal-cache" , create_if_missing = True
)
mount_path = "/root/.cache/torch"

auth_scheme = HTTPBearer()

def create_vertical_video(tracks , scores ,  pyframes_path , pyavi_path , audio_path , output_path, framerate=25):
    # Create vertical video using the provided tracks and audio
    target_width = 1080
    target_height = 1920

    flist = glob.glob(os.path.join(pyframes_path, "*.jpg"))
    flist.sort()
    faces = [[] for _ in range(len(flist))]

    for tidx, track in enumerate(tracks):
        score_array = scores[tidx]
        for fidx, frame in enumerate(track["track"]["frame"].tolist()):
            slice_start = max(fidx - 30 , 0)
            slice_end = min(fidx + 30 , len(score_array))
            score_slice =score_array[slice_start:slice_end]
            avg_score = float(np.mean(score_slice) 
                              if len(score_slice) > 0 else 0)
            faces[frame].append({
                'track':tidx, 'score':avg_score, 's':track['proc_track']["s"][fidx],'x':track['proc_track']["x"][fidx],'y':track['proc_track']["y"][fidx] 
            })
    
    temp_video_path = os.path.join(pyavi_path, "video_only.mp4")

    vout = None
    for fidx , frame in tqdm(enumerate(flist),total = len(flist) , desc="Creating vertical video"):
        img = cv2.imread(frame)
        if img is None:
            continue
        
        current_faces = faces[fidx]
        max_score_face = max(current_faces, key=lambda face: face['score']) if current_faces else None

        if max_score_face and max_score_face['score'] < 0:
            max_score_face = None

        if vout is None:
            vout = ffmpegcv.videoWriterNV(
                file=temp_video_path,
                codec=None,
                fps = framerate,
                resize = (target_width, target_height),
            )

        if max_score_face:
            mode = "crop"
        else:
            mode = "resize"
        
        if mode == "resize":
            scale = target_width / img.shape[1]
            resized_h


def process_clip(base_dir:str , original_video_path:str , s3_key:str ,  start_time:float , end_time:float , clip_index:int , transcribe_segments:list):
    clip_name = f"clip_{clip_index}"
    s3_key_dir = os.path.dirname(s3_key)
    output_s3_key = f"{s3_key_dir}/{clip_name}.mp4"
    print(f"Output s3 key: {output_s3_key}")

    clip_dir = base_dir / clip_name
    clip_dir.mkdir(parents=True , exist_ok=True)

    # Segment path: original clip from start to end 
    clip_segment_path = clip_dir / f"{clip_name}_segment.mp4"
    vertical_mp4_path = clip_dir / "pyav" / "video_out_vertical.mp4"
    subtitle_output_path = clip_dir / "pyav" / "video_with_subtitles.mp4"

    (clip_dir / "pywork").mkdir(exist_ok = True)
    pyframes_path = clip_dir / "pyframes"
    pyavi_path = clip_dir / "pyavi"
    audio_path = clip_dir / "pyavi"/"audio.wav"

    pyframes_path.mkdir(exist_ok=True)
    pyavi_path.mkdir(exist_ok=True)

    duration = end_time - start_time
    cut_command = (f"ffmpeg -i {original_video_path} --ss {start_time} -t {duration}"
                   f"{clip_segment_path}")
    subprocess.run(cut_command, shell=True, check=True , capture_output=True)

    extract_cmd= f"ffmpeg -i {clip_segment_path} -vn -acodec pcm_s16le -ar 16000 -ac 1 {audio_path}"
    subprocess.run(extract_cmd, shell=True, check=True , capture_output=True)

    shutil.copy(clip_segment_path , base_dir / f"{clip_name}.mp4")
    columbia_command = (f"python Columbia_test.py --videoName {clip_name}"
                        f" --videoFolder {str(base_dir)}"
                        f" --pretrainModel weight/finetuning_TalkSet.model")
    columbia_start_time = time.time()
    subprocess.run(columbia_command, cwd="/asd",shell=True)
    columbia_end_time = time.time()
    print(f"Columbia processing completed in {columbia_end_time - columbia_start_time:.2f} seconds.")

    tracks_path = clip_dir / "pywork" / "tracks.pckl"
    scores_path = clip_dir / "pywork" / "scores.pckl"
    if not tracks_path.exists() or not scores_path.exists():
        raise FileNotFoundError("Tracks or scores not found for clip")
    
    with open(tracks_path, "rb") as f:
        tracks = pickle.load(f)

    with open(scores_path ,"rb") as f:
        scores = pickle.load(f)

    cvv_start_time = time.time()
    create_vertical_video(
        tracks, scores , pyframes_path , pyavi_path , audio_path , vertical_mp4_path
    )
    cvv_end_time = time.time()
    print(f"Create vertical video completed in {cvv_end_time - cvv_start_time:.2f} seconds.")



@app.cls(gpu="A10G" , timeout = 900, retries=0, scaledown_window=20,secrets=[modal.Secret.from_name("ai-podcast-secret")], volumes={mount_path: volume})
class AiPodcastClipper:
    @modal.enter()
    def load_modal(self):
        
        print("Loading modals...")
        self.whisperx_model = whisperx.load_model("large-v2", device="cuda", compute_type="float16")
        self.alignment_model = whisperx.load_align_model(language_code="en",device="cuda")
        print("Transcription models loaded.")

        print("Creating gemini client ")
        self.gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        print("Gemini client created.")

    def transcribe_video(self , base_dir :str , video_path : str):
        audio_path = base_dir / "audio.wav"
        extract_cmd= f"ffmpeg -i {video_path} -vn -acodec pcm_s16le -ar 16000 -ac 1 {audio_path}"
        subprocess.run(extract_cmd, shell=True, check=True , capture_output=True)

        print("Starting transcription with whisperx...")
        start_time = time.time()

        audio = whisperx.load_audio(audio_path)
        result = self.whisperx_model.transcribe(audio, batch_size=16)

        result = whisperx.align(result["segments"], self.alignment_model, audio, audio_fs=16000, device="cuda",return_char_alignments=False)

        duration = time.time() - start_time
        print(f"Transcription completed in {duration:.2f} seconds.")

        segments = []
        if "word_segments" in result:
            for word_segment in result["word_segments"]:
                segments.append({
                    "start":word_segment("start"),
                    "end":word_segment("end"),
                    "word":word_segment("word")
                })
        
        return json.dumps(segments)
    
    def identify_moments(self , transcript:dict):
        response  = self.gemini_client.model.generate_content(model="Gemini-2.5-Flash-Lite",content="""
This is a podcast video transcript consisting of word, along with each words's start and end time. I am looking to create clips between a minimum of 30 and maximum of 60 seconds long. The clip should never exceed 60 seconds.

    Your task is to find and extract stories, or question and their corresponding answers from the transcript.
    Each clip should begin with the question and conclude with the answer.
    It is acceptable for the clip to include a few additional sentences before a question if it aids in contextualizing the question.

    Please adhere to the following rules:
    - Ensure that clips do not overlap with one another.
    - Start and end timestamps of the clips should align perfectly with the sentence boundaries in the transcript.
    - Only use the start and end timestamps provided in the input. modifying timestamps is not allowed.
    - Format the output as a list of JSON objects, each representing a clip with 'start' and 'end' timestamps: [{"start": seconds, "end": seconds}, ...clip2, clip3]. The output should always be readable by the python json.loads function.
    - Aim to generate longer clips between 40-60 seconds, and ensure to include as much content from the context as viable.

    Avoid including:
    - Moments of greeting, thanking, or saying goodbye.
    - Non-question and answer interactions.

    If there are no valid clips to extract, the output should be an empty list [], in JSON format. Also readable by json.loads() in Python.

    The transcript is as follows:\n\n                                                  
""" + str(transcript))
        print(f"Identifying moments in transcript...${response.text}")
        return response.text

   

    @modal.fastapi_endpoint(method="POST")
    def process_video(self , request: ProcessVideoRequest , token: HTTPAuthorizationCredentials = Depends(auth_scheme)):
        s3_key = request.s3_key
        if token.credentials != os.environ("AUTH_TOKEN"):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing token", headers={"WWW-Authenticate": "Bearer"})
        
        run_id = str(uuid.uuid4())
        base_dir = pathlib.Path("/tmp") / run_id
        base_dir.mkdir(parents=True, exist_ok=True)

        # Download the video file from S3
        video_path = base_dir / "input.mp4"
        s3_client = boto3.client("s3")
        s3_client.download_file("ai-podcast-clipper", s3_key, str(video_path))
        print(os.listdir(base_dir))

        # 1 Transcribe the video
        transcribe_segments_json = self.transcribe_video(base_dir, video_path)
        transcribe_segments = json.loads(transcribe_segments_json)

        # 2 Identify moments for clips
        print("Identifying moments for clips...")
        identified_moments_raw = self.identify_moments(transcribe_segments)
        cleaned_json_string = identified_moments_raw.strip()

        if(cleaned_json_string.startswith("```json")):
            cleaned_json_string = cleaned_json_string[len("```json"):].strip()
        if(cleaned_json_string.endswith("```")):
            cleaned_json_string = cleaned_json_string[:-len("```")].strip()

        clip_moments = json.loads(cleaned_json_string)
        if not clip_moments or not isinstance(clip_moments, list):
            print("Error identified moments is not a list")
            clip_moments = []

        print(f"Identified clip moments: {clip_moments}")

        # # Process each clip 
        for index , moment in enumerate(clip_moments[:3]):
            if("start" in moment and "end" in moment):
                print(f"Processing clip {index} from {moment['start']} to {moment['end']}")
                process_clip(base_dir , video_path , s3_key , moment["start"] , moment["end"] , index, transcribe_segments)

        if base_dir.exists():
            print(f"Cleaning up temp dir after {base_dir}")
            shutil.rmtree(base_dir , ignore_errors=True)

@app.local_entrypoint()
def main():
    import requests

    ai_podcast_clipper = AiPodcastClipper()

    url = ai_podcast_clipper.process_video.web_url

    payload = {
        "s3_key": "test1/pod5min.mp4"
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer 123123"
    }

    response = requests.post(url , json = payload, headers = headers)
    response.raise_for_status()
    result = response.json()
    print(result)


    