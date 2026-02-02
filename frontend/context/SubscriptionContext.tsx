import React, { createContext, useState, useContext, useEffect, ReactNode } from 'react';
import Purchases, { PurchasesOffering, CustomerInfo } from 'react-native-purchases';
import { Platform } from 'react-native';

const REVENUECAT_API_KEY = {
  ios: 'YOUR_IOS_API_KEY_HERE',
  android: 'YOUR_ANDROID_API_KEY_HERE',
};

interface SubscriptionContextType {
  offerings: PurchasesOffering | null;
  customerInfo: CustomerInfo | null;
  isLoading: boolean;
  isPremium: boolean;
  purchasePackage: (packageId: string) => Promise<void>;
  restorePurchases: () => Promise<void>;
}

const SubscriptionContext = createContext<SubscriptionContextType | undefined>(undefined);

export function SubscriptionProvider({ children }: { children: ReactNode }) {
  const [offerings, setOfferings] = useState<PurchasesOffering | null>(null);
  const [customerInfo, setCustomerInfo] = useState<CustomerInfo | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    initializePurchases();
  }, []);

  const initializePurchases = async () => {
    try {
      // Initialize RevenueCat
      const apiKey = Platform.OS === 'ios' ? REVENUECAT_API_KEY.ios : REVENUECAT_API_KEY.android;
      
      // Only initialize if not in web
      if (Platform.OS !== 'web') {
        await Purchases.configure({ apiKey });
        
        // Get current offerings
        const offerings = await Purchases.getOfferings();
        if (offerings.current) {
          setOfferings(offerings.current);
        }
        
        // Get customer info
        const info = await Purchases.getCustomerInfo();
        setCustomerInfo(info);
      }
    } catch (error) {
      console.error('Error initializing purchases:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const purchasePackage = async (packageId: string) => {
    try {
      if (!offerings) return;
      
      const packageToPurchase = offerings.availablePackages.find(
        (pkg) => pkg.identifier === packageId
      );
      
      if (packageToPurchase) {
        const { customerInfo } = await Purchases.purchasePackage(packageToPurchase);
        setCustomerInfo(customerInfo);
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
    try {
      const info = await Purchases.restorePurchases();
      setCustomerInfo(info);
    } catch (error) {
      console.error('Error restoring purchases:', error);
      throw error;
    }
  };

  const isPremium = customerInfo?.entitlements.active['premium'] !== undefined;

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
