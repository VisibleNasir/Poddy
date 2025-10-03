import { serve } from "inngest/next";
import { inngest } from "../../../inngest/client";
import { processVideo } from "../../../inngest/functions";

// Create an API that serves your Inngest functions
export const { GET, POST, PUT } = serve({
  client: inngest,
  functions: [
    processVideo
  ],
});