import React, { createContext, useContext, useState, useEffect } from 'react';
import { Platform } from 'react-native';

const REVENUECAT_ENABLED = !__DEV__;

let Purchases: any = null;
try {
  Purchases = require('react-native-purchases').default;
} catch (e) {
  console.log('RevenueCat not available');
}

interface SubscriptionContextType {
  offerings: any;
  purchasePackage: (pkg: any) => Promise<void>;
  isLoading: boolean;
  restorePurchases: () => Promise<void>;
  initializeRevenueCat: (userId?: string) => Promise<void>;
}

const SubscriptionContext = createContext<SubscriptionContextType | undefined>(undefined);

export function SubscriptionProvider({ children }: { children: React.ReactNode }) {
  const [offerings, setOfferings] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [initialized, setInitialized] = useState(false);

  const apiKey = Platform.OS === 'ios'
    ? 'appl_OVnBBsTafRUvxYPvVfFMfhuvEva'
    : 'goog_LSdTYjNzFKaMnhJQRcfEzGRwOmt';

  const initializeRevenueCat = async (userId?: string) => {
    if (!REVENUECAT_ENABLED || !Purchases || initialized) return;
    try {
      Purchases.configure({ apiKey });

      // CRITICAL: Log in with backend user ID so RevenueCat links subscriptions to our users
      if (userId) {
        try {
          const { customerInfo } = await Purchases.logIn(userId);
          console.log('[RC] Logged in as:', userId);
        } catch (loginErr) {
          console.warn('[RC] logIn failed, continuing as anonymous:', loginErr);
        }
      }

      setInitialized(true);
      const offerings = await Purchases.getOfferings();
      if (offerings.current) {
        setOfferings(offerings.current);
      }
    } catch (e) {
      console.log('RevenueCat init error:', e);
    }
  };

  useEffect(() => {
    initializeRevenueCat();
  }, []);

  const purchasePackage = async (pkg: any) => {
    if (!Purchases) throw new Error('Purchases not available');
    setIsLoading(true);
    try {
      const { customerInfo } = await Purchases.purchasePackage(pkg);
      if (customerInfo.entitlements.active['premium']) {
        console.log('[RC] Purchase successful - premium active');
      }
    } finally {
      setIsLoading(false);
    }
  };

  const restorePurchases = async () => {
    if (!Purchases) return;
    setIsLoading(true);
    try {
      await Purchases.restorePurchases();
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <SubscriptionContext.Provider value={{
      offerings,
      purchasePackage,
      isLoading,
      restorePurchases,
      initializeRevenueCat,
    }}>
      {children}
    </SubscriptionContext.Provider>
  );
}

export function useSubscription() {
  const context = useContext(SubscriptionContext);
  if (context === undefined) {
    throw new Error('useSubscription must be used within a SubscriptionProvider');
  }
  return context;
}
