import { requireNativeModule, Platform } from 'expo-modules-core';

let ExpoSubscriptionStore: any = null;

if (Platform.OS === 'ios') {
  try {
    ExpoSubscriptionStore = requireNativeModule('ExpoSubscriptionStore');
  } catch (e) {
    console.warn('ExpoSubscriptionStore not available:', e);
  }
}

export async function presentSubscriptionStore(groupID: string): Promise<void> {
  if (!ExpoSubscriptionStore) {
    throw new Error('SubscriptionStoreView is only available on iOS');
  }
  return ExpoSubscriptionStore.presentSubscriptionStore(groupID);
}

export function isSubscriptionStoreAvailable(): boolean {
  if (!ExpoSubscriptionStore) return false;
  try {
    return ExpoSubscriptionStore.isSubscriptionStoreAvailable();
  } catch {
    return false;
  }
}
