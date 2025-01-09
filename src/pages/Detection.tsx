import { useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { useNavigate } from "react-router-dom";
import { Camera, StopCircle } from "lucide-react";
import * as cocoSsd from "@tensorflow-models/coco-ssd";
import "@tensorflow/tfjs";

const Detection = () => {
  const navigate = useNavigate();
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isActive, setIsActive] = useState(false);
  const [model, setModel] = useState<cocoSsd.ObjectDetection | null>(null);
  const [videoLoaded, setVideoLoaded] = useState(false);

  useEffect(() => {
    const loadModel = async () => {
      const loadedModel = await cocoSsd.load();
      setModel(loadedModel);
      console.log("Model loaded");
    };
    loadModel();
  }, []);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          facingMode: "environment",
          width: { ideal: 640 },
          height: { ideal: 480 }
        } 
      });
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => {
          setVideoLoaded(true);
          if (canvasRef.current) {
            canvasRef.current.width = videoRef.current!.videoWidth;
            canvasRef.current.height = videoRef.current!.videoHeight;
          }
        };
        setIsActive(true);
      }
    } catch (err) {
      console.error("Error accessing camera:", err);
    }
  };

  const stopCamera = () => {
    if (videoRef.current?.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
      setIsActive(false);
      setVideoLoaded(false);
    }
  };

  useEffect(() => {
    let animationId: number;

    const detect = async () => {
      if (!model || !videoRef.current || !canvasRef.current || !isActive || !videoLoaded) return;

      // Ensure video is playing and has valid dimensions
      if (videoRef.current.readyState !== 4 || 
          videoRef.current.videoWidth === 0 || 
          videoRef.current.videoHeight === 0) {
        animationId = requestAnimationFrame(detect);
        return;
      }

      const predictions = await model.detect(videoRef.current);
      
      const ctx = canvasRef.current.getContext("2d");
      if (!ctx) return;

      ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

      predictions.forEach(prediction => {
        const [x, y, width, height] = prediction.bbox;
        
        ctx.strokeStyle = "#3B82F6";
        ctx.lineWidth = 2;
        ctx.strokeRect(x, y, width, height);

        ctx.fillStyle = "#3B82F6";
        ctx.fillRect(x, y - 25, prediction.class.length * 14, 25);
        
        ctx.fillStyle = "white";
        ctx.font = "18px Arial";
        ctx.fillText(prediction.class, x + 5, y - 5);
      });

      animationId = requestAnimationFrame(detect);
    };

    if (isActive && videoLoaded) {
      detect();
    }

    return () => {
      if (animationId) {
        cancelAnimationFrame(animationId);
      }
    };
  }, [model, isActive, videoLoaded]);

  return (
    <div className="min-h-screen p-4">
      <div className="max-w-2xl mx-auto">
        <Button
          variant="ghost"
          className="mb-4"
          onClick={() => {
            stopCamera();
            navigate("/");
          }}
        >
          ← Back
        </Button>

        <div className="relative aspect-video w-full overflow-hidden rounded-lg bg-gray-100">
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            className="w-full h-full object-cover"
          />
          <canvas
            ref={canvasRef}
            className="absolute inset-0 w-full h-full"
          />
          {!isActive && (
            <div className="absolute inset-0 flex items-center justify-center">
              <Button
                size="lg"
                className="text-xl"
                onClick={startCamera}
                disabled={!model}
              >
                <Camera className="mr-2 h-6 w-6" />
                {model ? "Start Camera" : "Loading Model..."}
              </Button>
            </div>
          )}
        </div>

        {isActive && (
          <Button
            variant="destructive"
            size="lg"
            className="mt-4 w-full text-xl"
            onClick={stopCamera}
          >
            <StopCircle className="mr-2 h-6 w-6" />
            Stop Camera
          </Button>
        )}
      </div>
    </div>
  );
};

export default Detection;