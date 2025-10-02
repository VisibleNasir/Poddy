import glob
import cv2
import ffmpegcv
import modal
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Depends, HTTPException, status
from pydantic import BaseModel
import os
import json
import pathlib
import subprocess
import time
import uuid
import boto3
from botocore.exceptions import ClientError, ParamValidationError
import re
import numpy as np
import shutil
import pysubs2
from tqdm import tqdm
import whisperx
import pickle
from google import genai


class ProcessVideoRequest(BaseModel):
    s3_key: str



image = (modal.Image.from_registry(
    "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.12")
    .apt_install(["ffmpeg", "libgl1-mesa-glx", "wget", "libcudnn8", "libcudnn8-dev"])
    .pip_install_from_requirements("requirements.txt")
    .run_commands(["mkdir -p /usr/share/fonts/truetype/custom",
                   "wget -O /usr/share/fonts/truetype/custom/Anton-Regular.ttf https://github.com/google/fonts/raw/main/ofl/anton/Anton-Regular.ttf",
                   "fc-cache -f -v"])
    .add_local_dir("asd", "/asd", copy=True))

app = modal.App("ai-podcast-clipper", image=image)

volume = modal.Volume.from_name(
    "ai-podcast-clipper-modal-cache", create_if_missing=True
)
mount_path = "/root/.cache/torch"

auth_scheme = HTTPBearer()

# S3 bucket name (use env var to avoid hardcoding and accidental typos/spaces)
_S3_BUCKET_RAW = os.environ.get("S3_BUCKET", "visible-podcast-clipper")


def _normalize_and_validate_bucket(bucket_raw: str) -> str:
    if bucket_raw is None:
        raise ValueError("S3_BUCKET is not set")
    bucket = bucket_raw.strip()
    if not re.match(r'^[a-zA-Z0-9.\-_]{1,255}$', bucket):
        raise ValueError(f'Invalid S3 bucket name "{bucket}"')
    return bucket


try:
    S3_BUCKET = _normalize_and_validate_bucket(_S3_BUCKET_RAW)
    print(f"Using S3 bucket: '{S3_BUCKET}'")
except ValueError as e:
    print(f"S3 bucket configuration error: {e}")
    S3_BUCKET = None


def create_vertical_video(tracks, scores, pyframes_path, pyavi_path, audio_path, output_path, framerate=25):
    target_width = 1080
    target_height = 1920

    flist = glob.glob(os.path.join(pyframes_path, "*.jpg"))
    flist.sort()
    print(f"Found {len(flist)} frames in {pyframes_path}")

    if not flist:
        print(f"No frames found in {pyframes_path}. Cannot create video.")
        raise ValueError("No valid frames to process")

    # Get audio duration
    try:
        probe = subprocess.run(
            f"ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {audio_path}",
            shell=True, capture_output=True, text=True, check=True
        )
        audio_duration = float(probe.stdout.strip())
        print(f"Audio duration: {audio_duration} seconds")
    except subprocess.CalledProcessError as e:
        print(f"Failed to probe audio duration: {e}")
        audio_duration = 0

    faces = [[] for _ in range(len(flist))]
    for tidx, track in enumerate(tracks):
        score_array = scores[tidx]
        for fidx, frame in enumerate(track["track"]["frame"].tolist()):
            if fidx >= len(flist):
                print(
                    f"Track frame index {fidx} exceeds frame count {len(flist)}. Skipping.")
                continue
            slice_start = max(fidx - 30, 0)
            slice_end = min(fidx + 30, len(score_array))
            score_slice = score_array[slice_start:slice_end]
            avg_score = float(np.mean(score_slice)
                              if len(score_slice) > 0 else 0)
            faces[frame].append({
                'track': tidx, 'score': avg_score,
                's': track['proc_track']["s"][fidx],
                'x': track['proc_track']["x"][fidx],
                'y': track['proc_track']["y"][fidx]
            })

    temp_video_path = os.path.join(pyavi_path, "video_only.mp4")
    vout = None
    valid_frames_written = 0

    # Handle single-frame case
    if len(flist) == 1:
        print("Single-frame video detected. Creating static video.")
        img = cv2.imread(flist[0])
        if img is None or img.size == 0 or img.shape[0] <= 0 or img.shape[1] <= 0:
            print(
                f"Invalid frame: {flist[0]} shape={img.shape if img is not None else 'None'}")
            raise ValueError("Single frame is invalid")

        vout = ffmpegcv.VideoWriterNV(
            file=temp_video_path,
            codec='h264_nvenc',
            fps=framerate,
            resize=(target_width, target_height),
        )

        current_faces = faces[0]
        max_score_face = max(
            current_faces, key=lambda face: face['score']) if current_faces else None
        if max_score_face and max_score_face['score'] < 0:
            max_score_face = None
        mode = "crop" if max_score_face else "resize"

        if mode == "resize":
            img_w, img_h = img.shape[1], img.shape[0]
            if img_w == 0 or img_h == 0:
                print(f"Invalid dimensions: {flist[0]} shape={img.shape}")
                raise ValueError("Invalid image dimensions")
            scale = target_width / img_w
            resized_height = int(img_h * scale)
            if resized_height <= 0:
                print(f"Invalid resized height: {resized_height}")
                raise ValueError("Computed resized height invalid")
            try:
                resized_image = cv2.resize(
                    img, (target_width, resized_height), interpolation=cv2.INTER_AREA)
            except cv2.error as e:
                print(f"OpenCV resize error on frame {flist[0]}: {e}")
                raise
            scale_for_bg = max(target_width / img_w, target_height / img_h)
            bg_width = int(img_w * scale_for_bg)
            bg_height = int(img_h * scale_for_bg)
            try:
                blurred_background = cv2.resize(img, (bg_width, bg_height))
                blurred_background = cv2.GaussianBlur(
                    blurred_background, (121, 121), 0)
                crop_x = (bg_width - target_width) // 2
                crop_y = (bg_height - target_height) // 2
                blurred_background = blurred_background[crop_y:crop_y +
                    target_height, crop_x:crop_x + target_width]
                center_y = (target_height - resized_height) // 2
                blurred_background[center_y:center_y +
                    resized_height, :] = resized_image
                frame = blurred_background
            except cv2.error as e:
                print(f"OpenCV background processing error: {e}")
                raise
        else:  # crop
            img_h = img.shape[0]
            if img_h == 0:
                print(f"Invalid height: {flist[0]} shape={img.shape}")
                raise ValueError("Invalid image height")
            scale = target_height / img_h
            if scale <= 0:
                print(f"Invalid scale: {scale} for frame {flist[0]}")
                raise ValueError("Invalid scale for crop")
            try:
                resized_image = cv2.resize(
                    img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            except cv2.error as e:
                print(f"OpenCV resize error on crop for frame {flist[0]}: {e}")
                raise
            frame_width = resized_image.shape[1]
            center_x = int(
                max_score_face["x"] * scale if max_score_face else frame_width // 2)
            top_x = max(0, min(center_x - target_width //
                        2, frame_width - target_width))
            frame = resized_image[0:target_height, top_x:top_x + target_width]

        num_frames = int(audio_duration * framerate)
        print(f"Writing {num_frames} frames for static video")
        for _ in tqdm(range(num_frames), desc="Creating static video"):
            vout.write(frame)
            valid_frames_written += 1
        vout.release()

    else:  # Multiple frames
        for fidx, fname in tqdm(enumerate(flist), total=len(flist), desc="Creating vertical video"):
            img = cv2.imread(fname)
            if img is None or img.size == 0 or img.shape[0] <= 0 or img.shape[1] <= 0:
                print(
                    f"Skipping invalid frame: {fname} shape={img.shape if img is not None else 'None'}")
                continue

            current_faces = faces[fidx]
            max_score_face = max(
                current_faces, key=lambda face: face['score']) if current_faces else None
            if max_score_face and max_score_face['score'] < 0:
                max_score_face = None

            if vout is None:
                vout = ffmpegcv.VideoWriterNV(
                    file=temp_video_path,
                    codec='h264_nvenc',
                    fps=framerate,
                    resize=(target_width, target_height),
                )

            if max_score_face:
                mode = "crop"
            else:
                mode = "resize"

            if mode == "resize":
                img_w = img.shape[1]
                img_h = img.shape[0]
                if img_w == 0 or img_h == 0:
                    print(
                        f"Invalid image dimensions for resize: {fname} shape={img.shape}")
                    continue
                scale = target_width / img_w
                resized_height = int(img_h * scale)
                if resized_height <= 0:
                    print(
                        f"Computed resized_height <= 0 for frame {fname}, skipping")
                    continue
                try:
                    resized_image = cv2.resize(
                        img, (target_width, resized_height), interpolation=cv2.INTER_AREA)
                except cv2.error as e:
                    print(f"OpenCV resize error on frame {fname}: {e}")
                    continue
                scale_for_bg = max(target_width / img_w, target_height / img_h)
                bg_width = int(img_w * scale_for_bg)
                bg_height = int(img_h * scale_for_bg)
                try:
                    if bg_width <= 0 or bg_height <= 0:
                        print(
                            f"Invalid background size bg_width={bg_width}, bg_height={bg_height} for frame {fname}")
                        continue
                    blurred_background = cv2.resize(img, (bg_width, bg_height))
                    blurred_background = cv2.GaussianBlur(
                        blurred_background, (121, 121), 0)
                    crop_x = (bg_width - target_width) // 2
                    crop_y = (bg_height - target_height) // 2
                    blurred_background = blurred_background[crop_y:crop_y +
                        target_height, crop_x:crop_x + target_width]
                    center_y = (target_height - resized_height) // 2
                    blurred_background[center_y:center_y +
                        resized_height, :] = resized_image
                    vout.write(blurred_background)
                    valid_frames_written += 1
                except cv2.error as e:
                    print(
                        f"OpenCV background processing error on frame {fname}: {e}")
                    continue
            else:  # crop
                img_h = img.shape[0]
                if img_h == 0:
                    print(
                        f"Invalid image height for crop: {fname} shape={img.shape}")
                    continue
                scale = target_height / img_h
                if scale <= 0:
                    print(
                        f"Invalid scale {scale} for crop on frame {fname}, skipping")
                    continue
                try:
                    resized_image = cv2.resize(
                        img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                except cv2.error as e:
                    print(
                        f"OpenCV resize error on crop for frame {fname}: {e}")
                    continue
                frame_width = resized_image.shape[1]
                center_x = int(
                    max_score_face["x"] * scale if max_score_face else frame_width // 2)
                top_x = max(0, min(center_x - target_width //
                            2, frame_width - target_width))
                image_cropped = resized_image[0:target_height,
                    top_x:top_x + target_width]
                vout.write(image_cropped)
                valid_frames_written += 1

        if vout:
            vout.release()

    if valid_frames_written == 0:
        print(f"No valid frames written to {temp_video_path}")
        raise ValueError("Failed to write any valid frames to video")

    print(f"Written {valid_frames_written} valid frames to {temp_video_path}")

    # Mux audio and video
    ffmpeg_command = (
        f"ffmpeg -y -i {temp_video_path} -i {audio_path} "
        f"-c:v copy -c:a aac -b:a 128k -shortest "
        f"{output_path}"
    )
    try:
        result = subprocess.run(
            ffmpeg_command, shell=True, check=True, text=True, capture_output=True)
        print(f"Video created successfully: {output_path}")
        print(f"FFmpeg output: {result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg muxing failed: {e.stderr}")
        raise


def create_subtitles_with_ffmpeg(transcript_segments: list, clip_start: float, clip_end: float, clip_video_path: str, output_path: str, max_words: int = 5):
    temp_dir = os.path.dirname(output_path)
    subtitle_path = os.path.join(temp_dir, "temp_subtitles.ass")

    clip_segments = [segment for segment in transcript_segments
                     if segment.get("start") is not None
                     and segment.get("end") is not None
                     and segment.get("end") > clip_start
                     and segment.get("start") < clip_end]

    subtitles = []
    current_words = []
    current_start = None
    current_end = None

    for segment in clip_segments:
        word = segment.get("word", "").strip()
        seg_start = segment.get("start")
        seg_end = segment.get("end")

        if not word or seg_start is None or seg_end is None:
            continue

        start_rel = max(0.0, seg_start - clip_start)
        end_rel = max(0.0, seg_end - clip_start)

        if end_rel <= 0:
            continue

        if not current_words:
            current_start = start_rel
            current_end = end_rel
            current_words = [word]
        elif len(current_words) >= max_words:
            subtitles.append(
                (current_start, current_end, ' '.join(current_words)))
            current_words = [word]
            current_start = start_rel
            current_end = end_rel
        else:
            current_words.append(word)
            current_end = end_rel

    # Append the last group of words if any
    if current_words:
        subtitles.append((current_start, current_end, ' '.join(current_words)))

    print(f"Generated {len(subtitles)} subtitle events: {subtitles}")

    subs = pysubs2.SSAFile()
    subs.info["WrapStyle"] = 0
    subs.info["ScaledBorderAndShadow"] = "yes"
    subs.info["PlayResX"] = 1080
    subs.info["PlayResY"] = 1920
    subs.info["ScriptType"] = "v4.00+"

    style_name = "Default"
    new_style = pysubs2.SSAStyle()
    new_style.fontname = "Anton"
    new_style.fontsize = 80  # Reduced to prevent overlap
    new_style.primarycolor = pysubs2.Color(255, 255, 255)
    new_style.outline = 2.0
    new_style.shadow = 2.0
    new_style.shadowcolor = pysubs2.Color(0, 0, 128)
    new_style.alignment = 2  # Bottom-center
    new_style.marginl = 50
    new_style.marginr = 50
    new_style.marginv = 100  # Increased to move subtitles higher
    new_style.spacing = 0.0

    subs.styles[style_name] = new_style
    for i, (start, end, text) in enumerate(subtitles):
        start_time = pysubs2.make_time(s=start)
        end_time = pysubs2.make_time(s=end)
        line = pysubs2.SSAEvent(
            start=start_time, end=end_time, text=text, style=style_name)
        subs.events.append(line)

    subs.save(subtitle_path)
    print(f"Saved subtitle file: {subtitle_path}")

    # Burn subtitles
    ffmpeg_cmd = (
        f"ffmpeg -y -i {clip_video_path} -vf \"ass={subtitle_path}\" "
        f"-c:v h264_nvenc -preset fast -crf 23 -c:a copy "
        f"{output_path}"
    )
    try:
        result = subprocess.run(ffmpeg_cmd, shell=True,
                                check=True, text=True, capture_output=True)
        print(f"Subtitles burned successfully: {output_path}")
        print(f"FFmpeg subtitle output: {result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg subtitle burning failed: {e.stderr}")
        raise


def process_clip(base_dir: str, original_video_path: str, s3_key: str, start_time: float, end_time: float, clip_index: int, transcript_segments: list):
    clip_name = f"clip_{clip_index}"
    s3_key_dir = os.path.dirname(s3_key)
    output_s3_key = f"{s3_key_dir}/{clip_name}.mp4"
    print(f"Output s3 key: {output_s3_key}")

    clip_dir = base_dir / clip_name
    clip_dir.mkdir(parents=True, exist_ok=True)

    clip_segment_path = clip_dir / f"{clip_name}_segment.mp4"
    vertical_mp4_path = clip_dir / "pyavi" / "video_out_vertical.mp4"
    subtitle_output_path = clip_dir / "pyavi" / "video_with_subtitles.mp4"

    (clip_dir / "pywork").mkdir(exist_ok=True)
    pyframes_path = clip_dir / "pyframes"
    pyavi_path = clip_dir / "pyavi"
    audio_path = clip_dir / "pyavi" / "audio.wav"

    pyframes_path.mkdir(exist_ok=True)
    pyavi_path.mkdir(exist_ok=True)

    duration = end_time - start_time

    cut_command = f"ffmpeg -i {original_video_path} -ss {start_time} -t {duration} {clip_segment_path}"
    subprocess.run(cut_command, shell=True, check=True,
                   capture_output=True, text=True)

    extract_cmd = f"ffmpeg -i {clip_segment_path} -vn -acodec pcm_s16le -ar 16000 -ac 1 {audio_path}"
    subprocess.run(extract_cmd, shell=True, check=True, capture_output=True)

    shutil.copy(clip_segment_path, base_dir / f"{clip_name}.mp4")
    columbia_command = (
        f"python Columbia_test.py --videoName {clip_name}"
        f" --videoFolder {str(base_dir)} "
        f" --pretrainModel weight/finetuning_TalkSet.model"
    )

    columbia_start_time = time.time()
    subprocess.run(columbia_command, cwd="/asd", shell=True)
    columbia_end_time = time.time()
    print(
        f"Columbia script processing completed in {columbia_end_time - columbia_start_time:.2f} seconds.")

    tracks_path = clip_dir / "pywork" / "tracks.pckl"
    scores_path = clip_dir / "pywork" / "scores.pckl"
    if not tracks_path.exists() or not scores_path.exists():
        raise FileNotFoundError("Tracks or scores not found for clip")

    with open(tracks_path, "rb") as f:
        tracks = pickle.load(f)

    with open(scores_path, "rb") as f:
        scores = pickle.load(f)

    cvv_start_time = time.time()
    create_vertical_video(
        tracks, scores, pyframes_path, pyavi_path, audio_path, vertical_mp4_path
    )
    cvv_end_time = time.time()
    print(
        f"Clip {clip_index} vertical video creation time: {cvv_end_time - cvv_start_time:.2f} seconds.")

    create_subtitles_with_ffmpeg(transcript_segments, start_time,
                                 end_time, vertical_mp4_path, subtitle_output_path, max_words=5)

    s3_client = boto3.client("s3")
    try:
        s3_client.upload_file(str(subtitle_output_path),
                              S3_BUCKET, output_s3_key)
    except ClientError as e:
        print(
            f"S3 upload failed for {subtitle_output_path} -> {S3_BUCKET}/{output_s3_key}: {e}")
        raise


@app.cls(gpu="L40S", timeout=900, retries=0, scaledown_window=20, secrets=[modal.Secret.from_name("ai-podcast-secret")], volumes={mount_path: volume})
class AiPodcastClipper:
    @modal.enter()
    def load_modal(self):
        print("Loading modals...")
        self.whisperx_model = whisperx.load_model(
            "large-v2", device="cuda", compute_type="float16")
        self.alignment_model, self.metadata = whisperx.load_align_model(
            language_code="en", device="cuda")
        print("Transcription models loaded.")
        print("Creating gemini client")
        self.gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        print("Gemini client created.")

    def transcribe_video(self, base_dir: str, video_path: str):
        audio_path = base_dir / "audio.wav"
        extract_cmd = f"ffmpeg -i {video_path} -vn -acodec pcm_s16le -ar 16000 -ac 1 {audio_path}"
        subprocess.run(extract_cmd, shell=True,
                       check=True, capture_output=True)

        print("Starting transcription with whisperx...")
        start_time = time.time()
        audio = whisperx.load_audio(str(audio_path))
        result = self.whisperx_model.transcribe(audio, batch_size=16)
        result = whisperx.align(result["segments"], self.alignment_model,
                                self.metadata, audio, device="cuda", return_char_alignments=False)
        duration = time.time() - start_time
        print(f"Transcription completed in {duration:.2f} seconds.")

        segments = []
        if "word_segments" in result:
            for word_segment in result["word_segments"]:
                segments.append({
                    "start": word_segment["start"],
                    "end": word_segment["end"],
                    "word": word_segment["word"]
                })
        return json.dumps(segments)

    def identify_moments(self, transcript: dict):
        response = self.gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents="""
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
        print(f"Identified moments in transcript...${response.text}")
        return response.text

    @modal.fastapi_endpoint(method="POST")
    def process_video(self, request: ProcessVideoRequest, token: HTTPAuthorizationCredentials = Depends(auth_scheme)):
        s3_key = request.s3_key
        if token.credentials != os.environ["AUTH_TOKEN"]:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                                detail="Invalid or missing token", headers={"WWW-Authenticate": "Bearer"})

        run_id = str(uuid.uuid4())
        base_dir = pathlib.Path("/tmp") / run_id
        base_dir.mkdir(parents=True, exist_ok=True)

        video_path = base_dir / "input.mp4"
        s3_client = boto3.client("s3")
        if not S3_BUCKET:
            raise HTTPException(
                status_code=500, detail="Server S3 configuration error: S3_BUCKET is not set or invalid")
        try:
            bucket_to_use = _normalize_and_validate_bucket(str(S3_BUCKET))
            print(
                f"Attempting S3 download: bucket={bucket_to_use}, key={s3_key}")
            s3_client.download_file(bucket_to_use, s3_key, str(video_path))
        except ParamValidationError as e:
            print(f"S3 parameter validation error for bucket={S3_BUCKET}: {e}")
            raise HTTPException(
                status_code=500, detail=f"Invalid S3 bucket name: {S3_BUCKET}")
        except ClientError as e:
            err_code = e.response.get("Error", {}).get(
                "Code") if hasattr(e, 'response') else None
            print(
                f"S3 download failed for key={s3_key} from bucket={S3_BUCKET}: {e}")
            if err_code in ("404", "NoSuchKey"):
                raise HTTPException(
                    status_code=404, detail="S3 object not found")
            if err_code in ("403", "AccessDenied"):
                raise HTTPException(
                    status_code=403, detail="Access denied to S3 object")
            raise HTTPException(
                status_code=500, detail=f"S3 download error: {e}")

        transcript_segments_json = self.transcribe_video(base_dir, video_path)
        transcript_segments = json.loads(transcript_segments_json)

        print("Identifying moments for clips...")
        identified_moments_raw = self.identify_moments(transcript_segments)
        cleaned_json_string = identified_moments_raw.strip()

        if cleaned_json_string.startswith("```json"):
            cleaned_json_string = cleaned_json_string[len("```json"):].strip()
        if cleaned_json_string.endswith("```"):
            cleaned_json_string = cleaned_json_string[:-len("```")].strip()

        clip_moments = json.loads(cleaned_json_string)
        if not clip_moments or not isinstance(clip_moments, list):
            print("Error identified moments is not a list")
            clip_moments = []

        print(f"Identified clip moments: {clip_moments}")

        for index, moment in enumerate(clip_moments[:3]):
            if "start" in moment and "end" in moment:
                print(
                    f"Processing clip {index} from {moment['start']} to {moment['end']}")
                process_clip(base_dir, video_path, s3_key,
                             moment["start"], moment["end"], index, transcript_segments)

        if base_dir.exists():
            print(f"Cleaning up temp dir after {base_dir}")
            shutil.rmtree(base_dir, ignore_errors=True)

   


@ app.local_entrypoint()
def main():
    import requests

    ai_podcast_clipper=AiPodcastClipper()
    url=ai_podcast_clipper.process_video.web_url

    payload={
        "s3_key": "test1/podcast1.mp4"
    }

    headers={
        "Content-Type": "application/json",
        "Authorization": "Bearer 123123"
    }

    response=requests.post(url, json=payload, headers=headers)
    response.raise_for_status()
    result=response.json()
    print(result)
