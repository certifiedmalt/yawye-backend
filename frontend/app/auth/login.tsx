import { useState } from 'react';
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  StyleSheet,
  KeyboardAvoidingView,
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
  const { purchaseSubscription, priceString, isLoading: purchaseLoading } = useSubscription();
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
    <KeyboardAvoidingView
      style={styles.container}
      behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
    >
      <ScrollView contentContainerStyle={styles.scrollContent}>
        <View style={styles.header}>
          <Ionicons name="scan-outline" size={80} color="#4CAF50" />
          <Text style={styles.title}>You Are What You Eat</Text>
          <Text style={styles.subtitle}>AI-Powered Ingredient Analysis</Text>
        </View>

        {/* Premium subscription — available without login */}
        <TouchableOpacity
          style={[styles.premiumButton, purchaseLoading && styles.premiumButtonDisabled]}
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
        </TouchableOpacity>
        <Text style={styles.premiumNote}>
          No account required. Auto-renews monthly. Cancel anytime.
        </Text>
        <View style={styles.legalRow}>
          <TouchableOpacity onPress={() => Linking.openURL('https://web-production-66c05.up.railway.app/terms-of-service')}>
            <Text style={styles.legalText}>Terms</Text>
          </TouchableOpacity>
          <Text style={styles.legalSep}>|</Text>
          <TouchableOpacity onPress={() => Linking.openURL('https://web-production-66c05.up.railway.app/privacy-policy')}>
            <Text style={styles.legalText}>Privacy</Text>
          </TouchableOpacity>
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
          />

          <Text style={styles.label}>Password</Text>
          <TextInput
            style={styles.input}
            placeholder="••••••••"
            placeholderTextColor="#666"
            value={password}
            onChangeText={setPassword}
            secureTextEntry
          />

          <TouchableOpacity
            style={styles.button}
            onPress={handleLogin}
            disabled={loading}
          >
            <Text style={styles.buttonText}>
              {loading ? 'Logging in...' : 'Login'}
            </Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={styles.forgotButton}
            onPress={() => router.push('/auth/forgot-password')}
          >
            <Text style={styles.forgotText}>Forgot your password?</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={styles.linkButton}
            onPress={() => router.push('/auth/register')}
          >
            <Text style={styles.linkText}>Don't have an account? Sign up</Text>
          </TouchableOpacity>
        </View>
      </ScrollView>
    </KeyboardAvoidingView>
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
