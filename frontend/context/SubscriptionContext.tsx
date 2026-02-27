import React, { createContext, useState, useContext, useEffect, ReactNode } from 'react';
import { Platform } from 'react-native';

// RevenueCat is ENABLED for production builds with real API keys.
const REVENUECAT_ENABLED = true;

interface SubscriptionContextType {
  offerings: any | null;
  customerInfo: any | null;
  isLoading: boolean;
  isPremium: boolean;
  purchasePackage: (packageId: string) => Promise<void>;
  restorePurchases: () => Promise<void>;
}

const SubscriptionContext = createContext<SubscriptionContextType | undefined>(undefined);

export function SubscriptionProvider({ children }: { children: ReactNode }) {
  const [offerings, setOfferings] = useState<any | null>(null);
  const [customerInfo, setCustomerInfo] = useState<any | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    if (REVENUECAT_ENABLED && Platform.OS !== 'web') {
      initializePurchases();
    }
  }, []);

  const initializePurchases = async () => {
    try {
      setIsLoading(true);
      // Only import and use RevenueCat when enabled
      const Purchases = require('react-native-purchases').default;
      const apiKey = Platform.OS === 'ios'
        ? 'appl_test_UVSbyqEktCmCDPUoYrrsUqkuWhk'  // TODO: Replace with iOS production key when ready
        : 'goog_sSuefaqGfyQKJvmIkN';

      await Purchases.configure({ apiKey });
      
      const offeringsResult = await Purchases.getOfferings();
      if (offeringsResult.current) {
        setOfferings(offeringsResult.current);
      }
      
      const info = await Purchases.getCustomerInfo();
      setCustomerInfo(info);
    } catch (error) {
      console.warn('RevenueCat init skipped or failed:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const purchasePackage = async (packageId: string) => {
    if (!REVENUECAT_ENABLED || !offerings) {
      console.warn('Purchases not available');
      return;
    }
    try {
      const Purchases = require('react-native-purchases').default;
      const packageToPurchase = offerings.availablePackages.find(
        (pkg: any) => pkg.identifier === packageId
      );
      if (packageToPurchase) {
        const { customerInfo: info } = await Purchases.purchasePackage(packageToPurchase);
        setCustomerInfo(info);
      }
    } catch (error: any) {
      if (error.userCancelled) {
        console.log('User cancelled purchase');
      } else {
        console.error('Error purchasing package:', error);
        throw error;
      }
    }
  };

  const restorePurchases = async () => {
    if (!REVENUECAT_ENABLED) {
      console.warn('Purchases not available');
      return;
    }
    try {
      const Purchases = require('react-native-purchases').default;
      const info = await Purchases.restorePurchases();
      setCustomerInfo(info);
    } catch (error) {
      console.error('Error restoring purchases:', error);
      throw error;
    }
  };

  const isPremium = false; // Default to free tier until RevenueCat is enabled

  return (
    <SubscriptionContext.Provider
      value={{
        offerings,
        customerInfo,
        isLoading,
        isPremium,
        purchasePackage,
        restorePurchases,
      }}
    >
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
