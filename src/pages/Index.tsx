import { Button } from "@/components/ui/button";
import { useNavigate } from "react-router-dom";
import { Camera } from "lucide-react";

const Index = () => {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen flex flex-col items-center justify-center p-4 space-y-8">
      <h1 className="text-4xl font-bold text-center">Vision Assistant</h1>
      <div className="flex flex-col gap-4 w-full max-w-md">
        <Button
          size="lg"
          className="h-16 text-xl"
          onClick={() => navigate("/detection")}
        >
          <Camera className="mr-2 h-6 w-6" />
          Start Detection
        </Button>
        <Button
          variant="outline"
          size="lg"
          className="h-16 text-xl"
          onClick={() => navigate("/profile")}
        >
          Profile Settings
        </Button>
      </div>
    </div>
  );
};

export default Index;