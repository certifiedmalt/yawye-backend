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
  const [configured, setConfigured] = useState(false);
  const [loggedInUserId, setLoggedInUserId] = useState<string | null>(null);

  const apiKey = Platform.OS === 'ios'
    ? 'appl_OVnBBsTafRUvxYPvVfFMfhuvEva'
    : 'goog_LSdTYjNzFKaMnhJQRcfEzGRwOmt';

  const initializeRevenueCat = async (userId?: string) => {
    if (!REVENUECAT_ENABLED || !Purchases) return;
    try {
      // Configure SDK only once
      if (!configured) {
        Purchases.configure({ apiKey });
        setConfigured(true);
        console.log('[RC] SDK configured');
      }

      // CRITICAL: Log in whenever we have a userId and haven't logged in as this user yet
      // This links RevenueCat subscriptions to our backend user accounts
      if (userId && loggedInUserId !== userId) {
        try {
          const { customerInfo } = await Purchases.logIn(userId);
          console.log('[RC] Logged in as:', userId, '- active entitlements:', Object.keys(customerInfo.entitlements.active));
          setLoggedInUserId(userId);
        } catch (loginErr) {
          console.warn('[RC] logIn failed, continuing as anonymous:', loginErr);
        }
      }

      const offeringsResult = await Purchases.getOfferings();
      if (offeringsResult.current) {
        setOfferings(offeringsResult.current);
      }
    } catch (e) {
      console.log('RevenueCat init error:', e);
    }
  };

  useEffect(() => {
    // Configure SDK on mount (anonymous), userId will be linked later via initializeRevenueCat(userId)
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
