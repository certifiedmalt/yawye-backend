import React from 'react';
import { Stack, useRouter } from 'expo-router';
import { StatusBar } from 'expo-status-bar';
import { View, Text, StyleSheet, TouchableOpacity, Platform } from 'react-native';
import { AuthProvider } from '../context/AuthContext';
import { SubscriptionProvider } from '../context/SubscriptionContext';
import { Ionicons } from '@expo/vector-icons';
import '../i18n/config';

// Safely configure notifications — wrapped to prevent crashes on some devices
try {
  const Notifications = require('expo-notifications');
  Notifications.setNotificationHandler({
    handleNotification: async () => ({
      shouldShowAlert: true,
      shouldPlaySound: false,
      shouldSetBadge: false,
    }),
  });
} catch (e) {
  console.warn('Notifications setup skipped:', e);
}

// Error boundary to prevent white-screen crashes
class ErrorBoundary extends React.Component<
  { children: React.ReactNode },
  { hasError: boolean; error: Error | null }
> {
  constructor(props: any) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error('App crash caught:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <View style={ebStyles.container}>
          <Text style={ebStyles.title}>Something went wrong</Text>
          <Text style={ebStyles.message}>
            The app encountered an error. Please try restarting.
          </Text>
          <TouchableOpacity
            style={ebStyles.button}
            onPress={() => this.setState({ hasError: false, error: null })}
          >
            <Text style={ebStyles.buttonText}>Try Again</Text>
          </TouchableOpacity>
        </View>
      );
    }
    return this.props.children;
  }
}

const ebStyles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0c0c0c',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 24,
  },
  title: {
    color: '#fff',
    fontSize: 22,
    fontWeight: 'bold',
    marginBottom: 12,
  },
  message: {
    color: '#aaa',
    fontSize: 16,
    textAlign: 'center',
    marginBottom: 24,
  },
  button: {
    backgroundColor: '#00e676',
    paddingHorizontal: 24,
    paddingVertical: 12,
    borderRadius: 8,
  },
  buttonText: {
    color: '#000',
    fontSize: 16,
    fontWeight: 'bold',
  },
});


// Custom back button with larger touch target (44x44pt minimum for accessibility)
function HeaderBackButton() {
  const router = useRouter();
  return (
    <TouchableOpacity
      onPress={() => router.back()}
      style={{
        width: 44,
        height: 44,
        alignItems: 'center',
        justifyContent: 'center',
        marginLeft: Platform.OS === 'ios' ? -8 : 0,
      }}
      hitSlop={{ top: 10, bottom: 10, left: 10, right: 10 }}
      data-testid="header-back-button"
    >
      <Ionicons name="arrow-back" size={24} color="#fff" />
    </TouchableOpacity>
  );
}


export default function RootLayout() {
  return (
    <ErrorBoundary>
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
              headerBackVisible: false,
              headerLeft: ({ canGoBack }) =>
                canGoBack ? (
                  <HeaderBackButton />
                ) : null,
            }}
          >
            <Stack.Screen name="index" options={{ headerShown: false }} />
            <Stack.Screen name="auth/login" options={{ title: 'Login', headerShown: false }} />
            <Stack.Screen name="auth/register" options={{ title: 'Register', headerShown: false }} />
            <Stack.Screen name="auth/forgot-password" options={{ title: 'Reset Password', headerShown: false }} />
            <Stack.Screen name="main" options={{ headerShown: false }} />
            <Stack.Screen name="scan" options={{ title: 'Scan Product' }} />
            <Stack.Screen name="result" options={{ title: 'Product Analysis' }} />
            <Stack.Screen name="assistant" options={{ title: 'Health Assistant' }} />
            <Stack.Screen name="achievements" options={{ title: 'Achievements' }} />
            <Stack.Screen name="quiz" options={{ title: 'Daily Quiz' }} />
            <Stack.Screen name="library" options={{ title: 'My Library' }} />
            <Stack.Screen name="about" options={{ title: 'About' }} />
          </Stack>
        </AuthProvider>
      </SubscriptionProvider>
    </ErrorBoundary>
  );
}
