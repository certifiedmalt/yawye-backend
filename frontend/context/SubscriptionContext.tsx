import React, { createContext, useContext, useState, useEffect } from 'react';
import { Platform } from 'react-native';

let useIAP: any = null;
try {
  const iapModule = require('expo-iap');
  useIAP = iapModule.useIAP;
} catch (e) {
  console.log('[IAP] expo-iap not available (expected on web)');
}

const PRODUCT_ID = 'yawye_premium_monthly';

interface SubscriptionContextType {
  purchaseSubscription: () => Promise<void>;
  restorePurchases: () => Promise<any[]>;
  isLoading: boolean;
  priceString: string | null;
  isConnected: boolean;
  purchaseSuccess: boolean;
}

const SubscriptionContext = createContext<SubscriptionContextType | undefined>(undefined);

function SubscriptionProviderInner({ children }: { children: React.ReactNode }) {
  const [priceString, setPriceString] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [purchaseSuccess, setPurchaseSuccess] = useState(false);

  const {
    connected,
    subscriptions,
    fetchProducts,
    requestPurchase,
    finishTransaction,
    getAvailablePurchases,
  } = useIAP({
    onPurchaseSuccess: async (purchase: any) => {
      try {
        console.log('[IAP] Purchase success:', purchase.productId);
        await finishTransaction({ purchase, isConsumable: false });
        setPurchaseSuccess(true);
        console.log('[IAP] Transaction finished');
      } catch (err) {
        console.warn('[IAP] finishTransaction error:', err);
      }
    },
    onPurchaseError: (error: any) => {
      console.warn('[IAP] Purchase error:', error);
    },
  });

  useEffect(() => {
    if (connected) {
      console.log('[IAP] Connected, fetching subscription products');
      fetchProducts({ skus: [PRODUCT_ID], type: 'subs' }).then(() => {
        console.log('[IAP] Subscription products fetched');
      }).catch((err: any) => {
        console.warn('[IAP] fetchProducts error:', err);
      });
    }
  }, [connected]);

  useEffect(() => {
    if (subscriptions && subscriptions.length > 0) {
      const p = subscriptions[0];
      const price = p.localizedPrice || p.price || null;
      if (price) {
        // Format price properly - avoid floating point display issues
        const formatted = typeof price === 'number' 
          ? `£${price.toFixed(2)}` 
          : String(price);
        setPriceString(`${formatted}/month`);
      }
      console.log('[IAP] Subscription loaded:', p.productId, 'price:', price);
    }
  }, [subscriptions]);

  const purchaseSubscription = async () => {
    setIsLoading(true);
    try {
      await requestPurchase({
        type: 'subs',
        request: {
          apple: { sku: PRODUCT_ID },
          google: { sku: PRODUCT_ID },
        },
      });
    } finally {
      setIsLoading(false);
    }
  };

  const restorePurchases = async () => {
    setIsLoading(true);
    try {
      await getAvailablePurchases();
      return [];
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <SubscriptionContext.Provider value={{
      purchaseSubscription,
      restorePurchases,
      isLoading,
      priceString,
      isConnected: connected,
      purchaseSuccess,
    }}>
      {children}
    </SubscriptionContext.Provider>
  );
}

function SubscriptionProviderFallback({ children }: { children: React.ReactNode }) {
  const value: SubscriptionContextType = {
    purchaseSubscription: async () => { throw new Error('IAP not available'); },
    restorePurchases: async () => [],
    isLoading: false,
    priceString: null,
    isConnected: false,
    purchaseSuccess: false,
  };

  return (
    <SubscriptionContext.Provider value={value}>
      {children}
    </SubscriptionContext.Provider>
  );
}

export function SubscriptionProvider({ children }: { children: React.ReactNode }) {
  if (!useIAP || Platform.OS === 'web') {
    return <SubscriptionProviderFallback>{children}</SubscriptionProviderFallback>;
  }
  return <SubscriptionProviderInner>{children}</SubscriptionProviderInner>;
}

export function useSubscription() {
  const context = useContext(SubscriptionContext);
  if (context === undefined) {
    throw new Error('useSubscription must be used within a SubscriptionProvider');
  }
  return context;
}
