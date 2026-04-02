import React, { createContext, useContext, useState, useEffect, useRef } from 'react';
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
  const configuredRef = useRef(false);
  const loggedInUserRef = useRef<string | null>(null);

  const apiKey = Platform.OS === 'ios'
    ? 'appl_qDwlqIUvHJHGuewqEExfpAgaCpw'
    : 'goog_sSuefaqGfyQKJvmIkNrWEyVElTx';

  const fetchOfferings = async () => {
    try {
      const result = await Purchases.getOfferings();
      console.log('[RC] Offerings fetched, current:', result?.current ? 'YES' : 'NO',
        'packages:', result?.current?.availablePackages?.length || 0);
      if (result?.current) {
        setOfferings(result.current);
        return true;
      }
    } catch (e) {
      console.warn('[RC] getOfferings failed:', e);
    }
    return false;
  };

  const initializeRevenueCat = async (userId?: string) => {
    if (!REVENUECAT_ENABLED || !Purchases) {
      console.log('[RC] Skipping init - enabled:', REVENUECAT_ENABLED, 'Purchases:', !!Purchases);
      return;
    }

    try {
      // Configure SDK only once (using ref to avoid async state race)
      if (!configuredRef.current) {
        console.log('[RC] Configuring SDK with key:', apiKey.substring(0, 8) + '...');
        Purchases.configure({ apiKey });
        configuredRef.current = true;
        console.log('[RC] SDK configured successfully');
      }

      // Log in when we have a userId
      if (userId && loggedInUserRef.current !== userId) {
        try {
          console.log('[RC] Logging in as:', userId);
          const { customerInfo } = await Purchases.logIn(userId);
          loggedInUserRef.current = userId;
          console.log('[RC] Logged in successfully. Active entitlements:',
            Object.keys(customerInfo?.entitlements?.active || {}));
        } catch (loginErr) {
          console.warn('[RC] logIn failed:', loginErr);
          // Continue anyway - offerings should still work for anonymous users
        }
      }

      // Fetch offerings
      await fetchOfferings();
    } catch (e) {
      console.error('[RC] Init error:', e);
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
