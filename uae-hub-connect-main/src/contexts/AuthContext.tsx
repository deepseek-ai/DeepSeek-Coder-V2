import { createContext, useContext, useState, ReactNode } from "react";

export type UserRole = "brand" | "vendor" | "driver" | "admin";

interface User {
  id: string;
  name: string;
  email: string;
  role: UserRole;
  avatar?: string;
  company?: string;
}

interface AuthContextType {
  user: User | null;
  isAuthenticated: boolean;
  login: (role: UserRole) => void;
  logout: () => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

// Mock users for demo purposes
const mockUsers: Record<UserRole, User> = {
  brand: {
    id: "brand-001",
    name: "Noon Operations",
    email: "ops@noon.com",
    role: "brand",
    company: "Noon",
    avatar: "🟡",
  },
  vendor: {
    id: "vendor-001",
    name: "Ahmed Al-Maktoum",
    email: "ahmed@deliveryservices.ae",
    role: "vendor",
    company: "Delivery Services LLC",
    avatar: "🚚",
  },
  driver: {
    id: "driver-001",
    name: "Mohammed Khan",
    email: "m.khan@driver.ae",
    role: "driver",
    avatar: "👨🏾",
  },
  admin: {
    id: "admin-001",
    name: "Sarah Ibrahim",
    email: "sarah@onehub.ae",
    role: "admin",
    company: "OneHubDeliOps",
    avatar: "🌐",
  },
};

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);

  const login = (role: UserRole) => {
    setUser(mockUsers[role]);
  };

  const logout = () => {
    setUser(null);
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        isAuthenticated: !!user,
        login,
        logout,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
}
