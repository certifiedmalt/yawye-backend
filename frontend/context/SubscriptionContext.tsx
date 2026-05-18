import React, { createContext, useContext, useState, useEffect, useRef } from 'react';
import { Platform } from 'react-native';

// Native IAP
let IAP: any = null;
try {
  IAP = require('react-native-iap');
} catch (e) {
  console.log('[IAP] react-native-iap not available');
}

const PRODUCT_IDS = Platform.select({
  ios: ['yawye_premium_monthly'],
  android: ['yawye_premium_monthly'],
  default: [],
}) as string[];

interface SubscriptionContextType {
  products: any[];
  purchaseSubscription: (productId?: string) => Promise<void>;
  isLoading: boolean;
  restorePurchases: () => Promise<void>;
  priceString: string | null;
  isConnected: boolean;
}

const SubscriptionContext = createContext<SubscriptionContextType | undefined>(undefined);

export function SubscriptionProvider({ children }: { children: React.ReactNode }) {
  const [products, setProducts] = useState<any[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [priceString, setPriceString] = useState<string | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const purchaseListenerRef = useRef<any>(null);

  useEffect(() => {
    if (!IAP || __DEV__) {
      console.log('[IAP] Skipping init - IAP:', !!IAP, 'DEV:', __DEV__);
      return;
    }

    const init = async () => {
      try {
        // Setup StoreKit 2 mode on iOS
        if (Platform.OS === 'ios') {
          await IAP.setup({ storekitMode: 'STOREKIT2_MODE' });
        }

        const connected = await IAP.initConnection();
        console.log('[IAP] Connection result:', connected);
        setIsConnected(true);

        // Fetch subscription products
        try {
          const subs = await IAP.getSubscriptions({ skus: PRODUCT_IDS });
          console.log('[IAP] Subscriptions loaded:', subs?.length || 0);
          if (subs && subs.length > 0) {
            setProducts(subs);
            // Extract price string
            const sub = subs[0];
            const price = sub.localizedPrice || sub.price || null;
            if (price) {
              setPriceString(`${price}/month`);
            }
          }
        } catch (subErr) {
          console.warn('[IAP] getSubscriptions failed:', subErr);
        }
      } catch (e) {
        console.error('[IAP] Init error:', e);
      }
    };

    init();

    // Listen for purchase updates
    if (IAP.purchaseUpdatedListener) {
      purchaseListenerRef.current = IAP.purchaseUpdatedListener(async (purchase: any) => {
        console.log('[IAP] Purchase updated:', purchase.productId);
        try {
          // Finish the transaction
          if (Platform.OS === 'ios') {
            await IAP.finishTransaction({ purchase, isConsumable: false });
          } else {
            // Android: acknowledge the purchase
            if (purchase.purchaseToken) {
              await IAP.acknowledgePurchaseAndroid({ token: purchase.purchaseToken });
            }
            await IAP.finishTransaction({ purchase, isConsumable: false });
          }
          console.log('[IAP] Transaction finished successfully');
        } catch (err) {
          console.warn('[IAP] finishTransaction error:', err);
        }
      });
    }

    return () => {
      if (purchaseListenerRef.current) {
        purchaseListenerRef.current.remove();
      }
      IAP?.endConnection();
    };
  }, []);

  const purchaseSubscription = async (productId?: string) => {
    if (!IAP) throw new Error('IAP not available');
    setIsLoading(true);
    try {
      const sku = productId || PRODUCT_IDS[0];

      if (Platform.OS === 'ios') {
        await IAP.requestSubscription({ sku });
      } else {
        // Android subscription request
        const sub = products.find((p: any) => p.productId === sku);
        if (sub?.subscriptionOfferDetails?.length > 0) {
          await IAP.requestSubscription({
            sku,
            subscriptionOffers: [{
              sku,
              offerToken: sub.subscriptionOfferDetails[0].offerToken,
            }],
          });
        } else {
          await IAP.requestSubscription({ sku });
        }
      }
    } finally {
      setIsLoading(false);
    }
  };

  const restorePurchases = async () => {
    if (!IAP) return;
    setIsLoading(true);
    try {
      const purchases = await IAP.getAvailablePurchases();
      console.log('[IAP] Restored purchases:', purchases?.length || 0);
      return purchases;
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <SubscriptionContext.Provider value={{
      products,
      purchaseSubscription,
      isLoading,
      restorePurchases,
      priceString,
      isConnected,
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
