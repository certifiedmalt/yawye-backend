import { View, Text, StyleSheet, TouchableOpacity, Platform, Linking, Alert } from 'react-native';
import { useRouter } from 'expo-router';
import { useAuth } from '../context/AuthContext';
import { useSubscription } from '../context/SubscriptionContext';
import { Ionicons } from '@expo/vector-icons';

export default function ManageSubscription() {
  const { user } = useAuth();
  const { restorePurchases, isLoading, priceString } = useSubscription();
  const router = useRouter();
  const isPremium = user?.subscription_tier === 'premium';

  const handleRestore = async () => {
    try {
      const purchases = await restorePurchases();
      if (purchases && purchases.length > 0) {
        Alert.alert('Restored', 'Your subscription has been restored successfully.');
      } else {
        Alert.alert('No Purchases Found', 'No active subscriptions were found for this account.');
      }
    } catch (e) {
      Alert.alert('Error', 'Could not restore purchases. Please try again.');
    }
  };

  const handleCancel = () => {
    const url = Platform.OS === 'ios'
      ? 'https://apps.apple.com/account/subscriptions'
      : 'https://play.google.com/store/account/subscriptions';
    Linking.openURL(url);
  };

  return (
    <View style={styles.container}>
      <View style={styles.card}>
        <View style={styles.planHeader}>
          <Ionicons name={isPremium ? 'star' : 'star-outline'} size={28} color="#FFD700" />
          <View style={styles.planInfo}>
            <Text style={styles.planName}>YAWYE Premium</Text>
            <Text style={styles.planType}>Monthly Subscription</Text>
          </View>
        </View>

        <View style={styles.statusRow}>
          <Text style={styles.statusLabel}>Status</Text>
          <View style={[styles.statusBadge, isPremium ? styles.activeBadge : styles.inactiveBadge]}>
            <Text style={[styles.statusText, isPremium ? styles.activeText : styles.inactiveText]}>
              {isPremium ? 'Active' : 'Inactive'}
            </Text>
          </View>
        </View>

        <View style={styles.detailRow}>
          <Text style={styles.detailLabel}>Price</Text>
          <Text style={styles.detailValue}>{priceString || '£1.99/month'}</Text>
        </View>

        <View style={styles.detailRow}>
          <Text style={styles.detailLabel}>Renewal</Text>
          <Text style={styles.detailValue}>{isPremium ? 'Auto-renews monthly' : 'Not subscribed'}</Text>
        </View>

        <View style={styles.detailRow}>
          <Text style={styles.detailLabel}>Features</Text>
          <Text style={styles.detailValue}>Unlimited scans, full analysis</Text>
        </View>
      </View>

      {isPremium && (
        <TouchableOpacity style={styles.cancelButton} onPress={handleCancel}>
          <Ionicons name="settings-outline" size={18} color="#ff6b6b" />
          <Text style={styles.cancelText}>Cancel or Change Subscription</Text>
        </TouchableOpacity>
      )}

      <Text style={styles.cancelNote}>
        {isPremium
          ? 'Cancellation is managed through your device settings. You will retain access until the end of your billing period.'
          : 'Subscribe from the home screen to get unlimited scans.'}
      </Text>

      <TouchableOpacity
        style={styles.restoreButton}
        onPress={handleRestore}
        disabled={isLoading}
      >
        <Text style={styles.restoreText}>
          {isLoading ? 'Restoring...' : 'Restore Purchases'}
        </Text>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0c0c0c',
    padding: 20,
  },
  card: {
    backgroundColor: '#1a1a1a',
    borderRadius: 16,
    padding: 24,
    borderWidth: 1,
    borderColor: '#333',
  },
  planHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 24,
    paddingBottom: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#222',
  },
  planInfo: {
    marginLeft: 12,
  },
  planName: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#FFD700',
  },
  planType: {
    fontSize: 14,
    color: '#888',
    marginTop: 2,
  },
  statusRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  statusLabel: {
    fontSize: 15,
    color: '#888',
  },
  statusBadge: {
    paddingHorizontal: 12,
    paddingVertical: 4,
    borderRadius: 12,
  },
  activeBadge: {
    backgroundColor: 'rgba(76, 175, 80, 0.2)',
  },
  inactiveBadge: {
    backgroundColor: 'rgba(136, 136, 136, 0.2)',
  },
  statusText: {
    fontSize: 13,
    fontWeight: '600',
  },
  activeText: {
    color: '#4CAF50',
  },
  inactiveText: {
    color: '#888',
  },
  detailRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  detailLabel: {
    fontSize: 15,
    color: '#888',
  },
  detailValue: {
    fontSize: 15,
    color: '#fff',
  },
  cancelButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: 24,
    padding: 16,
    backgroundColor: '#1a1a1a',
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#ff6b6b33',
  },
  cancelText: {
    color: '#ff6b6b',
    fontSize: 15,
    fontWeight: '600',
    marginLeft: 8,
  },
  cancelNote: {
    color: '#666',
    fontSize: 12,
    textAlign: 'center',
    marginTop: 12,
    lineHeight: 18,
  },
  restoreButton: {
    marginTop: 24,
    padding: 16,
    alignItems: 'center',
    backgroundColor: '#1a1a1a',
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#333',
  },
  restoreText: {
    color: '#4CAF50',
    fontSize: 15,
    fontWeight: '600',
  },
});
