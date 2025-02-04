import { useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { useNavigate } from "react-router-dom";
import { Camera, StopCircle, Volume2, VolumeX, Eye } from "lucide-react";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { useToast } from "@/hooks/use-toast";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

interface Person {
  distance: string;
  confidence: number;
}

interface DetectionResponse {
  persons: Person[];
  currency_value: string | null;
  frame_height: number;
  frame_width: number;
}

const Detection = () => {
  const navigate = useNavigate();
  const { toast } = useToast();
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isActive, setIsActive] = useState(false);
  const [videoLoaded, setVideoLoaded] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const instructionQueueRef = useRef<string[]>([]);
  const lastSpokenTimeRef = useRef(Date.now());
  const lastCurrencyTimeRef = useRef(Date.now());
  const [lastCurrencyValue, setLastCurrencyValue] = useState<string | null>(null);

  const speakInstruction = async (text: string) => {
    const MINIMUM_GAP = 3000;
    
    if (isSpeaking || (Date.now() - lastSpokenTimeRef.current) < MINIMUM_GAP) {
      if (!instructionQueueRef.current.includes(text)) {
        instructionQueueRef.current.push(text);
      }
      return;
    }

    try {
      setIsSpeaking(true);
      const response = await fetch("http://localhost:5000/speak", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          text,
          language: "en"
        }),
      });

      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

      const audioBlob = await response.blob();
      const audioUrl = URL.createObjectURL(audioBlob);
      const audio = new Audio(audioUrl);
      
      if (!isMuted) {
        try {
          await audio.play();
          lastSpokenTimeRef.current = Date.now();
          
          audio.onended = () => {
            URL.revokeObjectURL(audioUrl);
            setIsSpeaking(false);
            
            setTimeout(() => {
              if (instructionQueueRef.current.length > 0) {
                const nextInstruction = instructionQueueRef.current.shift();
                if (nextInstruction) {
                  speakInstruction(nextInstruction);
                }
              }
            }, 1000);
          };
        } catch (playError) {
          console.error("Audio playback error:", playError);
          setIsSpeaking(false);
        }
      } else {
        URL.revokeObjectURL(audioUrl);
        setIsSpeaking(false);
      }
    } catch (error) {
      console.error("Speech generation error:", error);
      setIsSpeaking(false);
    }
  };

  const detectFrame = async (videoElement: HTMLVideoElement) => {
    if (!videoElement || !canvasRef.current) return;
    
    const canvas = document.createElement('canvas');
    canvas.width = videoElement.videoWidth;
    canvas.height = videoElement.videoHeight;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    ctx.drawImage(videoElement, 0, 0);
    const base64Frame = canvas.toDataURL('image/jpeg');

    try {
      const response = await fetch('http://localhost:5000/detect_frame', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ frame: base64Frame }),
      });

      if (!response.ok) throw new Error('Frame detection failed');

      const data: DetectionResponse = await response.json();
      
      // Handle currency detection
      if (data.currency_value && data.currency_value !== lastCurrencyValue) {
        const currencyMessage = `Detected ${data.currency_value} rupees note`;
        if (Date.now() - lastCurrencyTimeRef.current > 5000) {
          speakInstruction(currencyMessage);
          setLastCurrencyValue(data.currency_value);
          lastCurrencyTimeRef.current = Date.now();
        }
      }

      // Generate instructions for detected persons
      if (data.persons.length > 0) {
        data.persons.forEach((person, index) => {
          const message = `Person ${index + 1} detected at ${person.distance}`;
          speakInstruction(message);
        });
      }

    } catch (error) {
      console.error('Frame detection error:', error);
    }
  };

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
    }
  };

  useEffect(() => {
    let animationId: number;
    
    const detect = async () => {
      if (!videoRef.current || !canvasRef.current || !isActive || !videoLoaded) return;

      if (videoRef.current.readyState !== 4) {
        animationId = requestAnimationFrame(detect);
        return;
      }

      await detectFrame(videoRef.current);
      animationId = requestAnimationFrame(detect);
    };

    if (isActive && videoLoaded) {
      detect();
    }

    return () => {
      if (animationId) {
        cancelAnimationFrame(animationId);
      }
      instructionQueueRef.current = [];
    };
  }, [isActive, videoLoaded]);

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
              onClick={() => setIsMuted(!isMuted)}
            >
              {isMuted ? 
                <VolumeX className="h-5 w-5 text-red-400" /> : 
                <Volume2 className="h-5 w-5 text-green-400" />
              }
            </Button>
          </div>
        </div>

        <Card className="bg-black/30 border-none shadow-xl backdrop-blur-sm">
          <CardHeader>
            <CardTitle className="text-white flex items-center gap-2">
              <Eye className="h-6 w-6 text-blue-400" />
              Vision Assistant
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
                  >
                    <Camera className="mr-2 h-6 w-6" />
                    Start Camera
                  </Button>
                </div>
              )}
            </div>
          </CardContent>
        </Card>

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