import { useState } from 'react';
import {
  View,
  Text,
  TextInput,
  Pressable,
  StyleSheet,
  Platform,
  Alert,
  ScrollView,
  Linking,
} from 'react-native';
import { useRouter } from 'expo-router';
import { useAuth } from '../../context/AuthContext';
import { useSubscription } from '../../context/SubscriptionContext';
import { Ionicons } from '@expo/vector-icons';

export default function Login() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const { login } = useAuth();
  const { purchaseSubscription, restorePurchases, priceString, isLoading: purchaseLoading } = useSubscription();
  const router = useRouter();

  const handleLogin = async () => {
    if (!email || !password) {
      Alert.alert('Error', 'Please fill in all fields');
      return;
    }

    setLoading(true);
    try {
      await login(email, password);
      router.replace('/main');
    } catch (error: any) {
      Alert.alert('Login Failed', error.message);
    } finally {
      setLoading(false);
    }
  };

  const handleSubscribeWithoutAccount = async () => {
    try {
      await purchaseSubscription();
      Alert.alert(
        'Welcome to Premium!',
        'Create a free account to sync your subscription across all your devices.',
        [
          { text: 'Create Account', onPress: () => router.push('/auth/register') },
          { text: 'Maybe Later', style: 'cancel' },
        ]
      );
    } catch (error: any) {
      if (!error?.userCancelled && error?.code !== 'E_USER_CANCELLED') {
        console.warn('Purchase error:', error);
      }
    }
  };

  return (
    <ScrollView
      style={styles.container}
      contentContainerStyle={styles.scrollContent}
      keyboardShouldPersistTaps="handled"
      bounces={false}
    >
      <View style={styles.inner}>
        <View style={styles.header}>
          <Ionicons name="scan-outline" size={80} color="#4CAF50" />
          <Text style={styles.title}>You Are What You Eat</Text>
          <Text style={styles.subtitle}>AI-Powered Ingredient Analysis</Text>
        </View>

        {/* Premium subscription — available without login */}
        <Pressable
          style={({ pressed }) => [
            styles.premiumButton,
            purchaseLoading && styles.premiumButtonDisabled,
            pressed && { opacity: 0.8 },
          ]}
          onPress={handleSubscribeWithoutAccount}
          disabled={purchaseLoading}
        >
          <View style={styles.premiumContent}>
            <Ionicons name="star" size={20} color="#0c0c0c" />
            <View style={styles.premiumTextWrap}>
              <Text style={styles.premiumButtonText}>
                {purchaseLoading ? 'Processing...' : 'Get YAWYE Premium'}
              </Text>
              <Text style={styles.premiumPrice}>
                {priceString || '£1.99/month'} — Unlimited scans
              </Text>
            </View>
          </View>
        </Pressable>
        <Text style={styles.premiumNote}>
          No account required. Auto-renews monthly. Cancel anytime.
        </Text>
        <View style={styles.legalRow}>
          <Pressable onPress={() => Linking.openURL('https://web-production-66c05.up.railway.app/terms-of-service')}>
            <Text style={styles.legalText}>Terms</Text>
          </Pressable>
          <Text style={styles.legalSep}>|</Text>
          <Pressable onPress={() => Linking.openURL('https://web-production-66c05.up.railway.app/privacy-policy')}>
            <Text style={styles.legalText}>Privacy</Text>
          </Pressable>
          <Text style={styles.legalSep}>|</Text>
          <Pressable onPress={async () => {
            try {
              const purchases = await restorePurchases();
              if (purchases && purchases.length > 0) {
                Alert.alert('Restored', 'Your subscription has been restored.');
              } else {
                Alert.alert('No Purchases Found', 'No active subscriptions found.');
              }
            } catch (e) {
              Alert.alert('Error', 'Could not restore purchases.');
            }
          }}>
            <Text style={styles.legalText}>Restore Purchases</Text>
          </Pressable>
        </View>

        <View style={styles.divider}>
          <View style={styles.dividerLine} />
          <Text style={styles.dividerText}>or sign in</Text>
          <View style={styles.dividerLine} />
        </View>

        <View style={styles.form}>
          <Text style={styles.label}>Email</Text>
          <TextInput
            style={styles.input}
            placeholder="your@email.com"
            placeholderTextColor="#666"
            value={email}
            onChangeText={setEmail}
            autoCapitalize="none"
            keyboardType="email-address"
            data-testid="login-email-input"
          />

          <Text style={styles.label}>Password</Text>
          <TextInput
            style={styles.input}
            placeholder="••••••••"
            placeholderTextColor="#666"
            value={password}
            onChangeText={setPassword}
            secureTextEntry
            onSubmitEditing={handleLogin}
            data-testid="login-password-input"
          />

          <Pressable
            style={({ pressed }) => [
              styles.button,
              loading && { opacity: 0.5 },
              pressed && { opacity: 0.8 },
            ]}
            onPress={handleLogin}
            disabled={loading}
            data-testid="login-submit-button"
          >
            <Text style={styles.buttonText}>
              {loading ? 'Logging in...' : 'Login'}
            </Text>
          </Pressable>

          <Pressable
            style={({ pressed }) => [styles.forgotButton, pressed && { opacity: 0.7 }]}
            onPress={() => router.push('/auth/forgot-password')}
          >
            <Text style={styles.forgotText}>Forgot your password?</Text>
          </Pressable>

          <Pressable
            style={({ pressed }) => [styles.linkButton, pressed && { opacity: 0.7 }]}
            onPress={() => router.push('/auth/register')}
          >
            <Text style={styles.linkText}>Don't have an account? Sign up</Text>
          </Pressable>
        </View>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0c0c0c',
  },
  scrollContent: {
    flexGrow: 1,
    justifyContent: 'center',
    padding: 24,
  },
  inner: {
    maxWidth: 480,
    width: '100%',
    alignSelf: 'center',
  },
  header: {
    alignItems: 'center',
    marginBottom: 24,
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#fff',
    marginTop: 16,
  },
  subtitle: {
    fontSize: 16,
    color: '#888',
    marginTop: 8,
  },
  premiumButton: {
    backgroundColor: '#FFD700',
    borderRadius: 14,
    padding: 16,
    marginBottom: 8,
  },
  premiumButtonDisabled: {
    opacity: 0.6,
  },
  premiumContent: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  premiumTextWrap: {
    marginLeft: 12,
  },
  premiumButtonText: {
    color: '#0c0c0c',
    fontSize: 17,
    fontWeight: 'bold',
  },
  premiumPrice: {
    color: '#333',
    fontSize: 13,
    marginTop: 2,
  },
  premiumNote: {
    color: '#666',
    fontSize: 11,
    textAlign: 'center',
    marginBottom: 4,
  },
  legalRow: {
    flexDirection: 'row',
    justifyContent: 'center',
    marginBottom: 20,
  },
  legalText: {
    color: '#4CAF50',
    fontSize: 11,
    textDecorationLine: 'underline',
  },
  legalSep: {
    color: '#444',
    fontSize: 11,
    marginHorizontal: 6,
  },
  divider: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 20,
  },
  dividerLine: {
    flex: 1,
    height: 1,
    backgroundColor: '#333',
  },
  dividerText: {
    color: '#666',
    fontSize: 13,
    marginHorizontal: 12,
  },
  form: {
    width: '100%',
  },
  label: {
    color: '#fff',
    fontSize: 16,
    marginBottom: 8,
    fontWeight: '600',
  },
  input: {
    backgroundColor: '#1a1a1a',
    borderRadius: 12,
    padding: 16,
    color: '#fff',
    fontSize: 16,
    marginBottom: 24,
    borderWidth: 1,
    borderColor: '#333',
  },
  button: {
    backgroundColor: '#4CAF50',
    borderRadius: 12,
    padding: 16,
    alignItems: 'center',
    marginTop: 8,
  },
  buttonText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: 'bold',
  },
  linkButton: {
    marginTop: 24,
    alignItems: 'center',
  },
  linkText: {
    color: '#4CAF50',
    fontSize: 16,
  },
  forgotButton: {
    marginTop: 16,
    alignItems: 'center',
  },
  forgotText: {
    color: '#888',
    fontSize: 14,
  },
});
