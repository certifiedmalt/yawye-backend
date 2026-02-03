import { useState, useEffect, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ScrollView,
  RefreshControl,
  Alert,
  ActivityIndicator,
  Animated,
} from 'react-native';
import { useRouter } from 'expo-router';
import { useAuth } from '../context/AuthContext';
import { Ionicons } from '@expo/vector-icons';
import { SafeAreaView } from 'react-native-safe-area-context';
import axios from 'axios';

const BACKEND_URL = process.env.EXPO_PUBLIC_BACKEND_URL;


export default function Main() {
  const { user, logout, refreshUser, token } = useAuth();
  const router = useRouter();
  const [refreshing, setRefreshing] = useState(false);
  const [gamificationLoading, setGamificationLoading] = useState(false);
  const [gamificationError, setGamificationError] = useState<string | null>(null);
  const [gamification, setGamification] = useState<any | null>(null);

  const onRefresh = async () => {
    setRefreshing(true);
  useEffect(() => {
    const fetchGamification = async () => {
      if (!token) return;
      try {
        setGamificationLoading(true);
        setGamificationError(null);
        const response = await axios.get(`${BACKEND_URL}/api/gamification/stats`, {
          headers: { Authorization: `Bearer ${token}` },
        });
        setGamification(response.data);
      } catch (error: any) {
        console.error('Error loading gamification stats', error?.response?.data || error?.message);
        setGamificationError('Unable to load streak and XP right now.');
      } finally {
        setGamificationLoading(false);
      }
    };

    fetchGamification();
  }, [token]);

    await refreshUser();
    setRefreshing(false);
  };

  const handleLogout = () => {
    Alert.alert('Logout', 'Are you sure you want to logout?', [
      { text: 'Cancel', style: 'cancel' },
      {
        text: 'Logout',
        style: 'destructive',
        onPress: async () => {
          await logout();
          router.replace('/auth/login');
        },
      },
    ]);
  };

  const scansRemaining = user?.subscription_tier === 'premium' ? '∞' : (5 - (user?.daily_scans || 0));

  return (
    <SafeAreaView style={styles.container} edges={['top']}>
      <ScrollView
        contentContainerStyle={styles.scrollContent}
        refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} />}
      >
        <View style={styles.header}>
          <View>
            <Text style={styles.greeting}>Hello, {user?.name}!</Text>
            <Text style={styles.subtitle}>What are you eating today?</Text>
          </View>
          <TouchableOpacity onPress={handleLogout}>
            <Ionicons name="log-out-outline" size={28} color="#fff" />
          </TouchableOpacity>
        </View>

        <View style={styles.statsCard}>
          <View style={styles.statItem}>
            <Ionicons name="scan-outline" size={32} color="#4CAF50" />
            <Text style={styles.statValue}>{scansRemaining}</Text>
            <Text style={styles.statLabel}>Scans Remaining Today</Text>
          </View>
          <View style={styles.statDivider} />
          <View style={styles.statItem}>
            <Ionicons
              name={user?.subscription_tier === 'premium' ? 'star' : 'star-outline'}
              size={32}
              color="#FFD700"
            />
            <Text style={styles.statValue}>
              {user?.subscription_tier === 'premium' ? 'Premium' : 'Free'}
            </Text>
            <Text style={styles.statLabel}>Subscription</Text>
          </View>
        </View>

        <View style={styles.gamificationCard}>
          <View style={styles.gamificationHeader}>
            <Text style={styles.gamificationTitle}>Your Health Streak</Text>
            {gamificationLoading && <ActivityIndicator size="small" color="#4CAF50" />}
          </View>
          {gamificationError ? (
            <Text style={styles.gamificationError}>{gamificationError}</Text>
          ) : gamification ? (
            <>
              <View style={styles.gamificationRow}>
                <View style={styles.gamificationStat}>
                  <Ionicons name="flame" size={24} color="#FF5722" />
                  <Text style={styles.gamificationStatValue}>
                    {gamification.current_streak || 0} days
                  </Text>
                  <Text style={styles.gamificationStatLabel}>Current streak</Text>
                </View>
                <View style={styles.gamificationStat}>
                  <Ionicons name="star" size={24} color="#FFD700" />
                  <Text style={styles.gamificationStatValue}>
                    L{gamification.level || 1}
                  </Text>
                  <Text style={styles.gamificationStatLabel}>
                    {gamification.xp || 0} XP
                  </Text>
                </View>
              </View>
              <View style={styles.gamificationActions}>
                <TouchableOpacity
                  style={styles.gamificationButton}
                  onPress={() => router.push('/quiz')}
                >
                  <Ionicons name="school" size={18} color="#fff" />
                  <Text style={styles.gamificationButtonText}>Daily Quiz</Text>
                </TouchableOpacity>
                <TouchableOpacity
                  style={[styles.gamificationButton, { backgroundColor: '#2196F3' }]}
                  onPress={() => router.push('/achievements')}
                >
                  <Ionicons name="trophy" size={18} color="#fff" />
                  <Text style={styles.gamificationButtonText}>Achievements</Text>
                </TouchableOpacity>
              </View>
            </>
          ) : (
            <Text style={styles.gamificationError}>
              Start scanning to build your streak and earn XP.
            </Text>
          )}
        </View>

        <TouchableOpacity style={styles.scanButton} onPress={() => router.push('/scan')}>
          <Ionicons name="scan" size={48} color="#fff" />
          <Text style={styles.scanButtonText}>Scan a Product</Text>
        </TouchableOpacity>

        <TouchableOpacity style={styles.assistantButton} onPress={() => router.push('/assistant')}>
          <Ionicons name="chatbubbles" size={36} color="#fff" />
          <Text style={styles.assistantButtonText}>Health Assistant</Text>
          <Text style={styles.assistantSubtext}>Ask nutrition questions</Text>
        </TouchableOpacity>

        <View style={styles.features}>
          <Text style={styles.featuresTitle}>What We Analyze</Text>
          
          <View style={styles.featureItem}>
            <Ionicons name="warning" size={24} color="#FF5252" />
            <View style={styles.featureText}>
              <Text style={styles.featureTitle}>Harmful Ingredients</Text>
              <Text style={styles.featureDescription}>
                Identify ingredients that may pose health risks with scientific references
              </Text>
            </View>
          </View>

          <View style={styles.featureItem}>
            <Ionicons name="checkmark-circle" size={24} color="#4CAF50" />
            <View style={styles.featureText}>
              <Text style={styles.featureTitle}>Beneficial Ingredients</Text>
              <Text style={styles.featureDescription}>
                Discover healthy ingredients backed by research studies
              </Text>
            </View>
          </View>

          <View style={styles.featureItem}>
            <Ionicons name="analytics" size={24} color="#2196F3" />
            <View style={styles.featureText}>
              <Text style={styles.featureTitle}>Overall Health Score</Text>
              <Text style={styles.featureDescription}>
                Get a comprehensive health rating from 1-10
              </Text>
            </View>
          </View>
        </View>

        {user?.subscription_tier !== 'premium' && (
          <View style={styles.upgradeCard}>
            <Text style={styles.upgradeTitle}>Upgrade to Premium</Text>
            <Text style={styles.upgradeText}>
              • Unlimited scans per day{"\n"}
              • Detailed ingredient analysis{"\n"}
              • Save favorites{"\n"}
              • Scan history
            </Text>
            <TouchableOpacity style={styles.upgradeButton}>
              <Text style={styles.upgradeButtonText}>Upgrade Now - $4.99/month</Text>
            </TouchableOpacity>
          </View>
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0c0c0c',
  },
  scrollContent: {
    padding: 24,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 32,
  },
  greeting: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#fff',
  },
  subtitle: {
    fontSize: 16,
    color: '#888',
    marginTop: 4,
  },
  statsCard: {
    backgroundColor: '#1a1a1a',
    borderRadius: 16,
    padding: 24,
    flexDirection: 'row',
    marginBottom: 24,
  },
  statItem: {
    flex: 1,
    alignItems: 'center',
  },
  statDivider: {
    width: 1,
    backgroundColor: '#333',
    marginHorizontal: 16,
  },
  statValue: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#fff',
    marginTop: 8,
  },
  statLabel: {
    fontSize: 12,
    color: '#888',
    marginTop: 4,
    textAlign: 'center',
  },
  scanButton: {
    backgroundColor: '#4CAF50',
    borderRadius: 16,
    padding: 32,
    alignItems: 'center',
    marginBottom: 16,
  },
  scanButtonText: {
    color: '#fff',
    fontSize: 20,
    fontWeight: 'bold',
    marginTop: 12,
  },
  assistantButton: {
    backgroundColor: '#2196F3',
    borderRadius: 16,
    padding: 24,
    alignItems: 'center',
    marginBottom: 32,
  },
  assistantButtonText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: 'bold',
    marginTop: 8,
  },
  assistantSubtext: {
    color: '#E3F2FD',
    fontSize: 14,
    marginTop: 4,
  },
  features: {
    marginBottom: 24,
  },
  featuresTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#fff',
    marginBottom: 16,
  },
  featureItem: {
    flexDirection: 'row',
    backgroundColor: '#1a1a1a',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
  },
  featureText: {
    flex: 1,
    marginLeft: 16,
  },
  featureTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#fff',
    marginBottom: 4,
  },
  featureDescription: {
    fontSize: 14,
    color: '#888',
  },
  upgradeCard: {
    backgroundColor: '#1a1a1a',
    borderRadius: 16,
    padding: 24,
    borderWidth: 2,
    borderColor: '#FFD700',
  },
  upgradeTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#FFD700',
    marginBottom: 12,
  },
  upgradeText: {
    fontSize: 16,
    color: '#fff',
    lineHeight: 24,
    marginBottom: 16,
  },
  upgradeButton: {
    backgroundColor: '#FFD700',
    borderRadius: 12,
    padding: 16,
    alignItems: 'center',
  },
  upgradeButtonText: {
    color: '#000',
    fontSize: 16,
    fontWeight: 'bold',
  },
  gamificationCard: {
    backgroundColor: '#1a1a1a',
    borderRadius: 16,
    padding: 20,
    marginBottom: 24,
    borderWidth: 1,
    borderColor: '#333',
  },
  gamificationHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  gamificationTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#fff',
  },
  gamificationRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 16,
  },
  gamificationStat: {
    flex: 1,
    alignItems: 'center',
  },
  gamificationStatValue: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#fff',
    marginTop: 6,
  },
  gamificationStatLabel: {
    fontSize: 12,
    color: '#888',
    marginTop: 2,
  },
  gamificationActions: {
    flexDirection: 'row',
    gap: 12,
  },
  gamificationButton: {
    flex: 1,
    backgroundColor: '#4CAF50',
    borderRadius: 999,
    paddingVertical: 10,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
  },
  gamificationButtonText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: '600',
    marginLeft: 6,
  },
  gamificationError: {
    fontSize: 13,
    color: '#888',
  },
});
