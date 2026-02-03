import { Stack } from 'expo-router';
import { StatusBar } from 'expo-status-bar';
import { AuthProvider } from '../context/AuthContext';
import { SubscriptionProvider } from '../context/SubscriptionContext';
import '../i18n/config';

export default function RootLayout() {
  return (
    <SubscriptionProvider>
      <AuthProvider>
        <StatusBar style="light" />
        <Stack
          screenOptions={{
            headerStyle: {
              backgroundColor: '#1a1a1a',
            },
            headerTintColor: '#fff',
            headerTitleStyle: {
              fontWeight: 'bold',
            },
          }}
        >
          <Stack.Screen name="index" options={{ headerShown: false }} />
          <Stack.Screen name="auth/login" options={{ title: 'Login', headerShown: false }} />
          <Stack.Screen name="auth/register" options={{ title: 'Register', headerShown: false }} />
          <Stack.Screen name="main" options={{ headerShown: false }} />
          <Stack.Screen name="scan" options={{ title: 'Scan Product' }} />
          <Stack.Screen name="result" options={{ title: 'Product Analysis' }} />
          <Stack.Screen name="assistant" options={{ title: 'Health Assistant' }} />
        </Stack>
      </AuthProvider>
    </SubscriptionProvider>
  );
}
