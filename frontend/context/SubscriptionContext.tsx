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

// Inner component that uses the hook
function SubscriptionProviderInner({ children }: { children: React.ReactNode }) {
  const [priceString, setPriceString] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [purchaseSuccess, setPurchaseSuccess] = useState(false);

  const {
    connected,
    products,
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
      console.log('[IAP] Connected, fetching products');
      fetchProducts([PRODUCT_ID]).then(() => {
        console.log('[IAP] Products fetched');
      }).catch((err: any) => {
        console.warn('[IAP] fetchProducts error:', err);
      });
    }
  }, [connected]);

  useEffect(() => {
    if (products && products.length > 0) {
      const p = products[0];
      const price = p.localizedPrice || p.price || null;
      if (price) {
        setPriceString(`${price}/month`);
      }
      console.log('[IAP] Product loaded:', p.productId, 'price:', price);
    }
  }, [products]);

  const purchaseSubscription = async () => {
    setIsLoading(true);
    try {
      await requestPurchase({ productId: PRODUCT_ID });
    } finally {
      setIsLoading(false);
    }
  };

  const restorePurchases = async () => {
    setIsLoading(true);
    try {
      const purchases = await getAvailablePurchases();
      console.log('[IAP] Restored purchases:', purchases?.length || 0);
      return purchases || [];
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

// Fallback for web/environments without IAP
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
