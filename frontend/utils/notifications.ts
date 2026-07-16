import * as Notifications from 'expo-notifications';
import AsyncStorage from '@react-native-async-storage/async-storage';
import Constants from 'expo-constants';
import axios from 'axios';
import { Platform } from 'react-native';

const BACKEND_URL = process.env.EXPO_PUBLIC_BACKEND_URL || 'https://web-production-66c05.up.railway.app';

export async function scheduleDailyReminder() {
  if (Platform.OS === 'android') {
    await Notifications.setNotificationChannelAsync('daily-reminders', {
      name: 'Daily Reminders',
      importance: Notifications.AndroidImportance.HIGH,
      sound: 'default',
      vibrationPattern: [0, 250, 250, 250],
    });
  }
  await Notifications.cancelAllScheduledNotificationsAsync();
  await Notifications.scheduleNotificationAsync({
    content: {
      title: 'Shopping today?',
      body: "Scan before you buy — know what's really in your food.",
      ...(Platform.OS === 'android' ? { channelId: 'daily-reminders' } : {}),
    },
    trigger: { type: 'daily', hour: 18, minute: 0 } as any,
  });
}

export async function registerPushToken(authToken: string | null) {
  try {
    if (!authToken) return;
    const tokenData = await Notifications.getExpoPushTokenAsync({
      projectId: Constants.expoConfig?.extra?.eas?.projectId,
    });
    if (tokenData.data) {
      await axios.post(
        `${BACKEND_URL}/api/auth/push-token`,
        { push_token: tokenData.data },
        { headers: { Authorization: `Bearer ${authToken}` } }
      );
    }
  } catch (e) {
    console.log('Push token registration error:', e);
  }
}

// Ask for notification permission right after the user's first successful scan —
// the moment of highest engagement. Only ever asks once.
export async function askNotificationsAfterFirstScan(authToken: string | null) {
  try {
    const asked = await AsyncStorage.getItem('notifications_v2_scheduled');
    if (asked === 'true') return;
    const { status } = await Notifications.requestPermissionsAsync();
    await AsyncStorage.setItem('notifications_v2_scheduled', 'true');
    if (status !== 'granted') return;
    await scheduleDailyReminder();
    await registerPushToken(authToken);
  } catch (e) {
    console.log('Notification setup error', e);
  }
}
