import { Button } from "@/components/ui/button";
import { useNavigate } from "react-router-dom";
import { Camera, Eye } from "lucide-react";

const Index = () => {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen flex flex-col items-center justify-center p-4 space-y-8 bg-gradient-to-br from-blue-500/20 via-purple-500/20 to-pink-500/20 backdrop-blur-sm">
      <div className="w-full max-w-md p-8 rounded-2xl bg-white/80 backdrop-blur-md shadow-xl space-y-8">
        <div className="flex items-center justify-center space-x-4">
          <Eye className="w-12 h-12 text-blue-600 animate-pulse" />
          <h1 className="text-4xl font-bold text-center bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
            Vision Assistant
          </h1>
        </div>
        
        <div className="flex flex-col gap-4">
          <Button
            size="lg"
            className="h-16 text-xl bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 transition-all duration-300 shadow-lg hover:shadow-xl"
            onClick={() => navigate("/detection")}
          >
            <Camera className="mr-2 h-6 w-6" />
            Start Detection
          </Button>
          
          <Button
            variant="outline"
            size="lg"
            className="h-16 text-xl border-2 hover:bg-gradient-to-r hover:from-blue-100 hover:to-purple-100 transition-all duration-300"
            onClick={() => navigate("/profile")}
          >
            Profile Settings
          </Button>
        </div>
      </div>
    </div>
  );
};

export default Index;