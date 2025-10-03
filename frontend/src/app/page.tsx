import Link from "next/link";

export default function HomePage() {
  return (
    <div className="h-screen w-screen ">
      <h1>Visible Podcast Clipper</h1>

      <div className="bg-zinc-100 w-1/2 p-4 rounded-lg">
        <h3>Upload Video </h3>

        <div>
          <p>Select a video file to upload:</p>
          <input type="file" />
          <button className="bg-blue-500 text-white p-2 rounded">Upload</button>
        </div>
      </div>
    </div>
  );
}
