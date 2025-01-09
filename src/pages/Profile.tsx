import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useToast } from "@/components/ui/use-toast";
import { useNavigate } from "react-router-dom";

const Profile = () => {
  const [name, setName] = useState("John Doe");
  const [phone, setPhone] = useState("+1234567890");
  const [language, setLanguage] = useState("en");
  const { toast } = useToast();
  const navigate = useNavigate();

  const handleSave = () => {
    // In a real app, this would save to backend
    toast({
      title: "Profile Updated",
      description: "Your settings have been saved successfully.",
    });
  };

  return (
    <div className="min-h-screen p-4">
      <div className="max-w-md mx-auto space-y-6">
        <Button
          variant="ghost"
          className="mb-4"
          onClick={() => navigate("/")}
        >
          ← Back
        </Button>

        <h1 className="text-3xl font-bold">Profile Settings</h1>

        <div className="space-y-4">
          <div className="space-y-2">
            <label className="text-lg font-medium">Name</label>
            <Input
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="h-12 text-lg"
            />
          </div>

          <div className="space-y-2">
            <label className="text-lg font-medium">Phone Number</label>
            <Input
              value={phone}
              onChange={(e) => setPhone(e.target.value)}
              className="h-12 text-lg"
            />
          </div>

          <div className="space-y-2">
            <label className="text-lg font-medium">Preferred Language</label>
            <Select value={language} onValueChange={setLanguage}>
              <SelectTrigger className="h-12 text-lg">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="en">English</SelectItem>
                <SelectItem value="hi">Hindi</SelectItem>
                <SelectItem value="es">Spanish</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <Button
            size="lg"
            className="w-full h-12 text-lg mt-6"
            onClick={handleSave}
          >
            Save Changes
          </Button>
        </div>
      </div>
    </div>
  );
};

export default Profile;