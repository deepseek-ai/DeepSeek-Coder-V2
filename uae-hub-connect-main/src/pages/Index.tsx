import { Sidebar } from "@/components/layout/Sidebar";
import { Header } from "@/components/layout/Header";
import { HeroSection } from "@/components/dashboard/HeroSection";
import { QuickStats, PerformanceStats } from "@/components/dashboard/QuickStats";
import { UAEMap } from "@/components/dashboard/UAEMap";
import { LiveActivityFeed } from "@/components/dashboard/LiveActivityFeed";
import { BrandOverview } from "@/components/dashboard/BrandOverview";
import { DriverLeaderboard } from "@/components/dashboard/DriverLeaderboard";
import { DeliveryChart } from "@/components/dashboard/DeliveryChart";

const Index = () => {
  return (
    <div className="min-h-screen bg-background">
      {/* Sidebar */}
      <Sidebar />

      {/* Main Content Area */}
      <div className="pl-64">
        <Header />

        <main className="p-8">
          {/* Hero Section */}
          <HeroSection />

          {/* Primary KPIs */}
          <section className="mb-8">
            <QuickStats />
          </section>

          {/* Map & Activity Feed */}
          <section className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
            <div className="lg:col-span-2">
              <UAEMap />
            </div>
            <div>
              <LiveActivityFeed />
            </div>
          </section>

          {/* Performance Stats */}
          <section className="mb-8">
            <PerformanceStats />
          </section>

          {/* Brands & Drivers */}
          <section className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
            <BrandOverview />
            <DriverLeaderboard />
          </section>

          {/* Charts */}
          <section className="mb-8">
            <DeliveryChart />
          </section>

          {/* Footer */}
          <footer className="mt-12 pt-8 border-t border-border/50 text-center">
            <p className="text-sm text-muted-foreground">
              <span className="font-display text-primary">OneHubDeliOps</span> • UAE's Unified Delivery Ecosystem
            </p>
            <p className="text-xs text-muted-foreground mt-1">
              Aligned with UAE D33 Agenda, AI Strategy 2031 & Net-Zero 2050
            </p>
          </footer>
        </main>
      </div>
    </div>
  );
};

export default Index;
