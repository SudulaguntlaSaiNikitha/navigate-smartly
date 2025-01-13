import { useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { useNavigate } from "react-router-dom";
import { Camera, StopCircle, Volume2, VolumeX, Languages, AlertTriangle } from "lucide-react";
import * as cocoSsd from "@tensorflow-models/coco-ssd";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { useToast } from "@/hooks/use-toast";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import "@tensorflow/tfjs";

interface Instruction {
  text: string;
  region: "left" | "center" | "right";
  distance: string;
}

const Detection = () => {
  const navigate = useNavigate();
  const { toast } = useToast();
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isActive, setIsActive] = useState(false);
  const [model, setModel] = useState<cocoSsd.ObjectDetection | null>(null);
  const [videoLoaded, setVideoLoaded] = useState(false);
  const [instructions, setInstructions] = useState<Instruction[]>([]);
  const [language, setLanguage] = useState(() => localStorage.getItem("language") || "en");

  const [isMuted, setIsMuted] = useState(false);
  const [personCount, setPersonCount] = useState(0);
  const [showTranslation, setShowTranslation] = useState(false);
  const [translatedText, setTranslatedText] = useState("");

  const speakInstruction = async (text: string) => {
    try {
      // Add a check to ensure the backend is reachable
      const response = await fetch("http://localhost:5000/speak", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          text,
          language
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const audioBlob = await response.blob();
      const audioUrl = URL.createObjectURL(audioBlob);
      const audio = new Audio(audioUrl);
      
      if (!isMuted) {
        try {
          await audio.play();
        } catch (playError) {
          console.error("Audio playback error:", playError);
          toast({
            title: "Playback Error",
            description: "Failed to play audio instruction",
            variant: "destructive",
          });
        }
      }

      audio.onended = () => URL.revokeObjectURL(audioUrl);
    } catch (error) {
      console.error("Speech generation error:", error);
      toast({
        title: "Backend Connection Error",
        description: "Failed to connect to speech service. Please ensure the backend server is running.",
        variant: "destructive",
      });
    }
  };

  const estimateDistance = (objectHeight: number, frameHeight: number): string => {
    const relativeSize = objectHeight / frameHeight;
    if (relativeSize > 0.5) return "Very Close (< 1m)";
    if (relativeSize > 0.3) return "Close (1-2m)";
    if (relativeSize > 0.15) return "Medium (2-4m)";
    return "Far (> 4m)";
  };

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
      toast({
        title: "Error",
        description: "Failed to access camera",
        variant: "destructive",
      });
    }
  };

  const stopCamera = () => {
    if (videoRef.current?.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
      setIsActive(false);
      setVideoLoaded(false);
      setInstructions([]);
    }
  };

  useEffect(() => {
    let animationId: number;
    let lastSpokenTime = 0;
    const SPEAK_COOLDOWN = 3000; // 3 seconds cooldown between voice instructions

    const detect = async () => {
      if (!model || !videoRef.current || !canvasRef.current || !isActive || !videoLoaded) return;

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

      // Draw region divisions
      const leftBoundary = ctx.canvas.width / 3;
      const rightBoundary = (2 * ctx.canvas.width) / 3;

      ctx.strokeStyle = "#3B82F6";
      ctx.setLineDash([5, 5]);
      ctx.beginPath();
      ctx.moveTo(leftBoundary, 0);
      ctx.lineTo(leftBoundary, ctx.canvas.height);
      ctx.moveTo(rightBoundary, 0);
      ctx.lineTo(rightBoundary, ctx.canvas.height);
      ctx.stroke();
      ctx.setLineDash([]);

      const newInstructions: Instruction[] = [];

      predictions.forEach(prediction => {
        const [x, y, width, height] = prediction.bbox;
        const objectCenterX = x + width / 2;
        const distance = estimateDistance(height, ctx.canvas.height);
        
        ctx.strokeStyle = "#3B82F6";
        ctx.lineWidth = 2;
        ctx.strokeRect(x, y, width, height);

        // Draw label with distance
        ctx.fillStyle = "#3B82F6";
        ctx.fillRect(x, y - 35, prediction.class.length * 14 + distance.length * 8, 35);
        
        ctx.fillStyle = "white";
        ctx.font = "18px Arial";
        ctx.fillText(`${prediction.class} - ${distance}`, x + 5, y - 10);

        const baseInstruction = distance === "Very Close (< 1m)" ? "CAUTION! " : "";
        let instruction = "";
        
        if (objectCenterX < leftBoundary) {
          instruction = `${baseInstruction}${prediction.class} on the left (${distance}), move to the center or right.`;
          newInstructions.push({
            text: instruction,
            region: "left",
            distance
          });
        } else if (objectCenterX > rightBoundary) {
          instruction = `${baseInstruction}${prediction.class} on the right (${distance}), move to the center or left.`;
          newInstructions.push({
            text: instruction,
            region: "right",
            distance
          });
        } else {
          instruction = `${baseInstruction}${prediction.class} in the center (${distance}), avoid or move left/right.`;
          newInstructions.push({
            text: instruction,
            region: "center",
            distance
          });
        }

        // Speak instruction if cooldown has passed
        const currentTime = Date.now();
        if (currentTime - lastSpokenTime > SPEAK_COOLDOWN) {
          speakInstruction(instruction);
          lastSpokenTime = currentTime;
        }
      });

      setInstructions(newInstructions);
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
  }, [model, isActive, videoLoaded, language]);

  const toggleMute = () => {
    setIsMuted(!isMuted);
  };

  const translateInstruction = async (text: string) => {
    try {
      const response = await fetch("http://localhost:5000/translate", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          text,
          target_language: language
        }),
      });

      if (!response.ok) throw new Error("Translation failed");

      const data = await response.json();
      setTranslatedText(data.translated_text);
      setShowTranslation(true);
    } catch (error) {
      console.error("Translation error:", error);
      toast({
        title: "Error",
        description: "Failed to translate instruction",
        variant: "destructive",
      });
    }
  };

  return (
    <div className="min-h-screen p-4 bg-gradient-to-b from-gray-900 via-blue-900 to-gray-900">
      <div className="max-w-4xl mx-auto space-y-6">
        <div className="flex items-center justify-between mb-6">
          <Button
            variant="ghost"
            className="text-white hover:text-blue-300 transition-colors"
            onClick={() => {
              stopCamera();
              navigate("/");
            }}
          >
            ← Back
          </Button>
          <div className="flex gap-2">
            <Button
              variant="outline"
              className="bg-white/10 hover:bg-white/20 transition-all"
              onClick={toggleMute}
            >
              {isMuted ? 
                <VolumeX className="h-5 w-5 text-red-400" /> : 
                <Volume2 className="h-5 w-5 text-green-400" />
              }
            </Button>
            <Button
              variant="outline"
              className="bg-white/10 hover:bg-white/20 transition-all"
              onClick={() => setShowTranslation(!showTranslation)}
            >
              <Languages className="h-5 w-5 text-blue-400" />
            </Button>
          </div>
        </div>

        <Card className="bg-black/30 border-none shadow-xl backdrop-blur-sm">
          <CardHeader>
            <CardTitle className="text-white flex items-center gap-2">
              <AlertTriangle className="h-6 w-6 text-yellow-400" />
              Navigation Assistant
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="relative aspect-video w-full overflow-hidden rounded-lg border-2 border-blue-500/30">
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
                <div className="absolute inset-0 flex items-center justify-center bg-black/70 backdrop-blur-sm">
                  <Button
                    size="lg"
                    className="text-xl bg-blue-600 hover:bg-blue-700 transition-colors"
                    onClick={startCamera}
                    disabled={!model}
                  >
                    <Camera className="mr-2 h-6 w-6" />
                    {model ? "Start Camera" : "Loading Model..."}
                  </Button>
                </div>
              )}
            </div>
          </CardContent>
        </Card>

        {instructions.length > 0 && (
          <div className="space-y-2 animate-fade-in">
            {instructions.map((instruction, index) => (
              <Alert 
                key={index} 
                variant="default" 
                className={`
                  border-2 backdrop-blur-sm transition-all
                  ${instruction.distance.includes("Very Close") 
                    ? "bg-red-900/50 border-red-500 animate-pulse" 
                    : instruction.distance.includes("Close")
                    ? "bg-orange-900/50 border-orange-500"
                    : "bg-blue-900/50 border-blue-500"
                  }
                  text-white
                `}
              >
                <AlertDescription className="text-lg flex items-center gap-2">
                  <div className="flex-1">
                    {instruction.text}
                    {showTranslation && translatedText && (
                      <div className="mt-2 text-gray-300 text-sm">
                        {translatedText}
                      </div>
                    )}
                  </div>
                  {instruction.distance.includes("Very Close") && (
                    <AlertTriangle className="h-6 w-6 text-red-400 animate-pulse" />
                  )}
                </AlertDescription>
              </Alert>
            ))}
          </div>
        )}

        {isActive && (
          <Button
            variant="destructive"
            size="lg"
            className="mt-6 w-full text-xl transition-all hover:bg-red-600"
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
