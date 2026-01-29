import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { AuthProvider, useAuth } from "@/contexts/AuthContext";
import Index from "./pages/Index";
import NotFound from "./pages/NotFound";
import Auth from "./pages/Auth";
import BrandDashboard from "./pages/brand/BrandDashboard";
import VendorDashboard from "./pages/vendor/VendorDashboard";
import DriverDashboard from "./pages/driver/DriverDashboard";
import AdminDashboard from "./pages/admin/AdminDashboard";

const queryClient = new QueryClient();

function ProtectedRoute({ children, allowedRole }: { children: React.ReactNode; allowedRole: string }) {
  const { user, isAuthenticated } = useAuth();
  
  if (!isAuthenticated) {
    return <Navigate to="/auth" replace />;
  }
  
  if (allowedRole && user?.role !== allowedRole) {
    return <Navigate to="/auth" replace />;
  }
  
  return <>{children}</>;
}

function AppRoutes() {
  const { isAuthenticated, user } = useAuth();

  return (
    <Routes>
      <Route path="/" element={
        isAuthenticated ? <Navigate to={`/${user?.role}`} replace /> : <Navigate to="/auth" replace />
      } />
      <Route path="/auth" element={
        isAuthenticated ? <Navigate to={`/${user?.role}`} replace /> : <Auth />
      } />
      <Route path="/brand" element={
        <ProtectedRoute allowedRole="brand"><BrandDashboard /></ProtectedRoute>
      } />
      <Route path="/vendor" element={
        <ProtectedRoute allowedRole="vendor"><VendorDashboard /></ProtectedRoute>
      } />
      <Route path="/driver" element={
        <ProtectedRoute allowedRole="driver"><DriverDashboard /></ProtectedRoute>
      } />
      <Route path="/admin" element={
        <ProtectedRoute allowedRole="admin"><AdminDashboard /></ProtectedRoute>
      } />
      <Route path="/legacy" element={<Index />} />
      <Route path="*" element={<NotFound />} />
    </Routes>
  );
}

const App = () => (
  <QueryClientProvider client={queryClient}>
    <AuthProvider>
      <TooltipProvider>
        <Toaster />
        <Sonner />
        <BrowserRouter>
          <AppRoutes />
        </BrowserRouter>
      </TooltipProvider>
    </AuthProvider>
  </QueryClientProvider>
);

export default App;
