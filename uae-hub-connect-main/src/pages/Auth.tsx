import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth, UserRole } from "@/contexts/AuthContext";
import { Globe, Building2, Truck, Users, Shield, ArrowRight, Zap } from "lucide-react";
import { cn } from "@/lib/utils";

interface RoleOption {
  id: UserRole;
  title: string;
  description: string;
  icon: React.ElementType;
  gradient: string;
  route: string;
}

const roleOptions: RoleOption[] = [
  {
    id: "brand",
    title: "Delivery Brand",
    description: "Noon, Amazon, Careem, Keeta, Porter",
    icon: Building2,
    gradient: "from-primary/20 to-primary/5",
    route: "/brand",
  },
  {
    id: "vendor",
    title: "Vendor Fleet",
    description: "Delivery services & logistics partners",
    icon: Truck,
    gradient: "from-cyan/20 to-cyan/5",
    route: "/vendor",
  },
  {
    id: "driver",
    title: "Driver",
    description: "Delivery partners & fleet riders",
    icon: Users,
    gradient: "from-success/20 to-success/5",
    route: "/driver",
  },
  {
    id: "admin",
    title: "OneHub Team",
    description: "Platform administration & oversight",
    icon: Shield,
    gradient: "from-destructive/20 to-destructive/5",
    route: "/admin",
  },
];

export default function Auth() {
  const [selectedRole, setSelectedRole] = useState<UserRole | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const { login } = useAuth();
  const navigate = useNavigate();

  const handleLogin = async () => {
    if (!selectedRole) return;
    
    setIsLoading(true);
    // Simulate authentication delay
    await new Promise((resolve) => setTimeout(resolve, 800));
    
    login(selectedRole);
    const route = roleOptions.find((r) => r.id === selectedRole)?.route || "/";
    navigate(route);
    setIsLoading(false);
  };

  return (
    <div className="min-h-screen bg-background flex">
      {/* Left Panel - Branding */}
      <div className="hidden lg:flex lg:w-1/2 relative overflow-hidden bg-gradient-to-br from-midnight via-card to-midnight">
        {/* Animated Background */}
        <div className="absolute inset-0">
          <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-primary/10 rounded-full blur-3xl animate-pulse" />
          <div className="absolute bottom-1/4 right-1/4 w-80 h-80 bg-cyan/10 rounded-full blur-3xl animate-pulse" style={{ animationDelay: "1s" }} />
        </div>
        
        {/* Content */}
        <div className="relative z-10 flex flex-col justify-center px-16">
          <div className="flex items-center gap-4 mb-8">
            <div className="w-16 h-16 rounded-2xl bg-primary/20 flex items-center justify-center glow-gold">
              <Globe className="w-8 h-8 text-primary" />
            </div>
            <div>
              <h1 className="font-display text-3xl font-bold text-gradient-gold">OneHubDeliOps</h1>
              <p className="text-muted-foreground">UAE's Unified Delivery Ecosystem</p>
            </div>
          </div>
          
          <h2 className="text-4xl font-display font-bold text-foreground leading-tight mb-6">
            Synchronizing the Future of
            <span className="text-gradient-gold block">UAE Logistics</span>
          </h2>
          
          <p className="text-lg text-muted-foreground mb-8 max-w-md">
            One platform connecting delivery brands, vendors, and drivers with sub-30ms latency synchronization.
          </p>
          
          <div className="grid grid-cols-2 gap-4">
            <div className="glass-card p-4">
              <p className="text-2xl font-display font-bold text-primary">3,847</p>
              <p className="text-sm text-muted-foreground">Active Deliveries</p>
            </div>
            <div className="glass-card p-4">
              <p className="text-2xl font-display font-bold text-cyan">847</p>
              <p className="text-sm text-muted-foreground">Online Drivers</p>
            </div>
            <div className="glass-card p-4">
              <p className="text-2xl font-display font-bold text-success">98.7%</p>
              <p className="text-sm text-muted-foreground">Success Rate</p>
            </div>
            <div className="glass-card p-4">
              <p className="text-2xl font-display font-bold text-foreground">28ms</p>
              <p className="text-sm text-muted-foreground">Sync Latency</p>
            </div>
          </div>
        </div>
      </div>
      
      {/* Right Panel - Login */}
      <div className="w-full lg:w-1/2 flex items-center justify-center p-8">
        <div className="w-full max-w-md">
          {/* Mobile Logo */}
          <div className="lg:hidden flex items-center gap-3 mb-8">
            <div className="w-12 h-12 rounded-xl bg-primary/20 flex items-center justify-center">
              <Globe className="w-6 h-6 text-primary" />
            </div>
            <div>
              <h1 className="font-display text-xl font-bold text-gradient-gold">OneHubDeliOps</h1>
              <p className="text-xs text-muted-foreground">UAE Logistics Hub</p>
            </div>
          </div>
          
          <div className="mb-8">
            <h2 className="text-2xl font-display font-bold text-foreground mb-2">Welcome Back</h2>
            <p className="text-muted-foreground">Select your portal to continue</p>
          </div>
          
          {/* Role Selection */}
          <div className="space-y-3 mb-8">
            {roleOptions.map((role, index) => (
              <button
                key={role.id}
                onClick={() => setSelectedRole(role.id)}
                className={cn(
                  "w-full flex items-center gap-4 p-4 rounded-xl border transition-all duration-300",
                  "hover:border-primary/50 hover:bg-secondary/50",
                  "animate-fade-in-up",
                  selectedRole === role.id
                    ? "border-primary bg-primary/10 ring-1 ring-primary/30"
                    : "border-border bg-card/50"
                )}
                style={{ animationDelay: `${index * 100}ms` }}
              >
                <div className={cn("w-12 h-12 rounded-xl flex items-center justify-center bg-gradient-to-br", role.gradient)}>
                  <role.icon className={cn(
                    "w-6 h-6",
                    selectedRole === role.id ? "text-primary" : "text-muted-foreground"
                  )} />
                </div>
                <div className="flex-1 text-left">
                  <p className={cn(
                    "font-medium",
                    selectedRole === role.id ? "text-foreground" : "text-foreground"
                  )}>{role.title}</p>
                  <p className="text-sm text-muted-foreground">{role.description}</p>
                </div>
                <div className={cn(
                  "w-5 h-5 rounded-full border-2 transition-all",
                  selectedRole === role.id
                    ? "border-primary bg-primary"
                    : "border-muted-foreground"
                )}>
                  {selectedRole === role.id && (
                    <div className="w-full h-full flex items-center justify-center">
                      <div className="w-2 h-2 rounded-full bg-primary-foreground" />
                    </div>
                  )}
                </div>
              </button>
            ))}
          </div>
          
          {/* Login Button */}
          <button
            onClick={handleLogin}
            disabled={!selectedRole || isLoading}
            className={cn(
              "w-full flex items-center justify-center gap-3 px-6 py-4 rounded-xl font-medium transition-all duration-300",
              selectedRole
                ? "bg-primary text-primary-foreground hover:bg-primary/90 glow-gold"
                : "bg-secondary text-muted-foreground cursor-not-allowed"
            )}
          >
            {isLoading ? (
              <>
                <Zap className="w-5 h-5 animate-pulse" />
                Authenticating...
              </>
            ) : (
              <>
                Continue to Portal
                <ArrowRight className="w-5 h-5" />
              </>
            )}
          </button>
          
          {/* UAE Pass Integration Note */}
          <div className="mt-6 p-4 rounded-xl bg-secondary/30 border border-border/50">
            <p className="text-xs text-muted-foreground text-center">
              🇦🇪 Production version integrates with <span className="text-primary font-medium">UAE Pass</span> & <span className="text-primary font-medium">Emirates ID</span> for secure authentication
            </p>
          </div>
          
          {/* Footer */}
          <p className="mt-8 text-center text-xs text-muted-foreground">
            Aligned with UAE D33 Agenda, AI Strategy 2031 & Net-Zero 2050
          </p>
        </div>
      </div>
    </div>
  );
}
