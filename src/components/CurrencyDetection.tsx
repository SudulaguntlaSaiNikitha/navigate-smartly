import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { useToast } from "@/hooks/use-toast";
import { IndianRupee, Upload } from "lucide-react";

interface CurrencyDetectionProps {
  onSpeak: (text: string) => void;
}

const CurrencyDetection = ({ onSpeak }: CurrencyDetectionProps) => {
  const { toast } = useToast();
  const [isDetecting, setIsDetecting] = useState(false);

  const handleCurrencyDetection = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setIsDetecting(true);
    const formData = new FormData();
    formData.append('image', file);

    try {
      const response = await fetch('http://localhost:5000/detect_currency', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) throw new Error('Currency detection failed');

      const data = await response.json();
      const message = `Detected ${data.currency_value} rupees note`;
      
      toast({
        title: "Currency Detected",
        description: message,
      });

      onSpeak(message);
    } catch (error) {
      console.error('Currency detection error:', error);
      toast({
        title: "Error",
        description: "Failed to detect currency",
        variant: "destructive",
      });
    } finally {
      setIsDetecting(false);
    }
  };

  return (
    <Card className="bg-black/30 border-none shadow-xl backdrop-blur-sm mt-4">
      <CardHeader>
        <CardTitle className="text-white flex items-center gap-2">
          <IndianRupee className="h-6 w-6 text-green-400" />
          Currency Detection
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex items-center gap-4">
          <Button
            variant="outline"
            className="bg-white/10 hover:bg-white/20 transition-all relative"
            disabled={isDetecting}
          >
            <input
              type="file"
              accept="image/*"
              onChange={handleCurrencyDetection}
              className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
            />
            <Upload className="h-5 w-5 text-blue-400 mr-2" />
            {isDetecting ? "Detecting..." : "Upload Currency Image"}
          </Button>
        </div>
      </CardContent>
    </Card>
  );
};

export default CurrencyDetection;