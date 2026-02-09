import { useState, useEffect, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ScrollView,
  Animated,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { useAuth } from '../context/AuthContext';
import { useRouter } from 'expo-router';
import axios from 'axios';

const BACKEND_URL = process.env.EXPO_PUBLIC_BACKEND_URL || 'https://web-production-66c05.up.railway.app';

interface GamificationStats {
  current_streak: number;
  longest_streak: number;
  total_scans: number;
  level: number;
  xp: number;
  badges: string[];
  daily_quests: any;
  quiz_streak: number;
  quiz_correct_answers: number;
  quiz_total_answers: number;
}

export default function Achievements() {
  const [stats, setStats] = useState<GamificationStats | null>(null);
  const [loading, setLoading] = useState(true);
  const { token } = useAuth();
  const router = useRouter();
  const scaleAnim = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    loadStats();
  }, []);

  useEffect(() => {
    if (stats) {
      Animated.spring(scaleAnim, {
        toValue: 1,
        friction: 4,
        useNativeDriver: true,
      }).start();
    }
  }, [stats]);

  const loadStats = async () => {
    try {
      const response = await axios.get(
        `${BACKEND_URL}/api/gamification/stats`,
        { headers: { Authorization: `Bearer ${token}` } }
      );
      setStats(response.data);
    } catch (error) {
      console.error('Error loading stats:', error);
    } finally {
      setLoading(false);
    }
  };

  const getLevelProgress = () => {
    if (!stats) return 0;
    const xpInCurrentLevel = stats.xp % 100;
    return xpInCurrentLevel;
  };

  const getNextLevelXP = () => {
    if (!stats) return 100;
    return ((Math.floor(stats.xp / 100) + 1) * 100);
  };

  const getBadgeName = (badgeId: string) => {
    const badges: { [key: string]: { name: string; icon: string } } = {
      streak_3: { name: '3-Day Warrior', icon: '🔥' },
      streak_7: { name: 'Week Champion', icon: '⭐' },
      streak_30: { name: 'Monthly Master', icon: '💎' },
      scanner_10: { name: 'Scanner Rookie', icon: '📱' },
      scanner_50: { name: 'Scan Expert', icon: '🎯' },
      scanner_100: { name: 'Scan Master', icon: '👑' },
    };
    return badges[badgeId] || { name: badgeId, icon: '🏆' };
  };

  const dailyQuestsCompleted = stats?.daily_quests
    ? Object.values(stats.daily_quests).filter((q: any) => q.completed).length
    : 0;
  
  const totalQuests = stats?.daily_quests ? Object.keys(stats.daily_quests).length : 0;

  return (
    <SafeAreaView style={styles.container} edges={['top', 'bottom']}>
      <ScrollView contentContainerStyle={styles.content}>
        <View style={styles.header}>
          <Ionicons name="trophy" size={48} color="#FFD700" />
          <Text style={styles.title}>Your Achievements</Text>
        </View>

        {/* Level Card */}
        <Animated.View style={[styles.levelCard, { transform: [{ scale: scaleAnim }] }]}>
          <View style={styles.levelHeader}>
            <View>
              <Text style={styles.levelLabel}>Level</Text>
              <Text style={styles.levelValue}>{stats?.level || 1}</Text>
            </View>
            <View style={styles.xpContainer}>
              <Ionicons name="star" size={20} color="#FFD700" />
              <Text style={styles.xpText}>{stats?.xp || 0} XP</Text>
            </View>
          </View>
          
          <View style={styles.progressBarContainer}>
            <View style={styles.progressBar}>
              <View
                style={[
                  styles.progressFill,
                  { width: `${getLevelProgress()}%` },
                ]}
              />
            </View>
            <Text style={styles.progressText}>
              {getLevelProgress()}/100 XP to Level {(stats?.level || 1) + 1}
            </Text>
          </View>
        </Animated.View>

        {/* Streaks */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>🔥 Streaks</Text>
          <View style={styles.streakRow}>
            <View style={styles.streakCard}>
              <Ionicons name="flame" size={32} color="#FF5722" />
              <Text style={styles.streakValue}>{stats?.current_streak || 0}</Text>
              <Text style={styles.streakLabel}>Current Streak</Text>
            </View>
            <View style={styles.streakCard}>
              <Ionicons name="trending-up" size={32} color="#FFD700" />
              <Text style={styles.streakValue}>{stats?.longest_streak || 0}</Text>
              <Text style={styles.streakLabel}>Longest Streak</Text>
            </View>
          </View>
        </View>

        {/* Daily Quests */}
        <View style={styles.section}>
          <View style={styles.sectionHeader}>
            <Text style={styles.sectionTitle}>📝 Daily Quests</Text>
            <Text style={styles.questProgress}>
              {dailyQuestsCompleted}/{totalQuests}
            </Text>
          </View>
          
          {stats?.daily_quests && Object.entries(stats.daily_quests).map(([key, quest]: [string, any]) => (
            <View key={key} style={styles.questCard}>
              <View style={styles.questIcon}>
                {quest.completed ? (
                  <Ionicons name="checkmark-circle" size={24} color="#4CAF50" />
                ) : (
                  <Ionicons name="ellipse-outline" size={24} color="#666" />
                )}
              </View>
              <View style={styles.questInfo}>
                <Text style={styles.questName}>
                  {key === 'scan_3_products' && 'Scan 3 Products'}
                  {key === 'find_healthy_product' && 'Find a Healthy Product (8+/10)'}
                  {key === 'use_assistant' && 'Use Health Assistant'}
                </Text>
                <Text style={styles.questReward}>+{quest.xp} XP</Text>
              </View>
            </View>
          ))}
        </View>

        {/* Stats */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>📊 Stats</Text>
          <View style={styles.statsGrid}>
            <View style={styles.statCard}>
              <Ionicons name="scan" size={24} color="#4CAF50" />
              <Text style={styles.statValue}>{stats?.total_scans || 0}</Text>
              <Text style={styles.statLabel}>Total Scans</Text>
            </View>
            <View style={styles.statCard}>
              <Ionicons name="school" size={24} color="#2196F3" />
              <Text style={styles.statValue}>{stats?.quiz_streak || 0}</Text>
              <Text style={styles.statLabel}>Quiz Streak</Text>
            </View>
            <View style={styles.statCard}>
              <Ionicons name="checkmark-done" size={24} color="#FFD700" />
              <Text style={styles.statValue}>
                {stats?.quiz_total_answers > 0
                  ? Math.round((stats.quiz_correct_answers / stats.quiz_total_answers) * 100)
                  : 0}%
              </Text>
              <Text style={styles.statLabel}>Quiz Accuracy</Text>
            </View>
          </View>
        </View>

        {/* Badges */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>🏆 Badges Earned</Text>
          {stats?.badges && stats.badges.length > 0 ? (
            <View style={styles.badgesGrid}>
              {stats.badges.map((badgeId: string, index: number) => {
                const badge = getBadgeName(badgeId);
                return (
                  <View key={index} style={styles.badgeCard}>
                    <Text style={styles.badgeIcon}>{badge.icon}</Text>
                    <Text style={styles.badgeName}>{badge.name}</Text>
                  </View>
                );
              })}
            </View>
          ) : (
            <Text style={styles.noBadges}>Complete challenges to earn badges!</Text>
          )}
        </View>

        {/* Action Buttons */}
        <View style={styles.actionsContainer}>
          <TouchableOpacity
            style={styles.actionButton}
            onPress={() => router.push('/quiz')}
          >
            <Ionicons name="school" size={24} color="#fff" />
            <Text style={styles.actionButtonText}>Daily Quiz</Text>
          </TouchableOpacity>
          
          <TouchableOpacity
            style={[styles.actionButton, { backgroundColor: '#2196F3' }]}
            onPress={() => router.push('/scan')}
          >
            <Ionicons name="scan" size={24} color="#fff" />
            <Text style={styles.actionButtonText}>Scan Product</Text>
          </TouchableOpacity>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0c0c0c',
  },
  content: {
    padding: 24,
  },
  header: {
    alignItems: 'center',
    marginBottom: 32,
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#fff',
    marginTop: 16,
  },
  levelCard: {
    backgroundColor: '#1a1a1a',
    borderRadius: 16,
    padding: 24,
    marginBottom: 24,
    borderWidth: 2,
    borderColor: '#FFD700',
  },
  levelHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  levelLabel: {
    fontSize: 14,
    color: '#888',
  },
  levelValue: {
    fontSize: 48,
    fontWeight: 'bold',
    color: '#FFD700',
  },
  xpContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#0a0a0a',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 20,
  },
  xpText: {
    color: '#FFD700',
    fontSize: 16,
    fontWeight: 'bold',
    marginLeft: 6,
  },
  progressBarContainer: {
    width: '100%',
  },
  progressBar: {
    height: 8,
    backgroundColor: '#333',
    borderRadius: 4,
    overflow: 'hidden',
    marginBottom: 8,
  },
  progressFill: {
    height: '100%',
    backgroundColor: '#FFD700',
  },
  progressText: {
    fontSize: 12,
    color: '#888',
  },
  section: {
    marginBottom: 32,
  },
  sectionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#fff',
    marginBottom: 16,
  },
  questProgress: {
    fontSize: 16,
    color: '#4CAF50',
    fontWeight: 'bold',
  },
  streakRow: {
    flexDirection: 'row',
    gap: 16,
  },
  streakCard: {
    flex: 1,
    backgroundColor: '#1a1a1a',
    borderRadius: 12,
    padding: 20,
    alignItems: 'center',
  },
  streakValue: {
    fontSize: 32,
    fontWeight: 'bold',
    color: '#fff',
    marginTop: 8,
  },
  streakLabel: {
    fontSize: 12,
    color: '#888',
    marginTop: 4,
  },
  questCard: {
    flexDirection: 'row',
    backgroundColor: '#1a1a1a',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    alignItems: 'center',
  },
  questIcon: {
    marginRight: 12,
  },
  questInfo: {
    flex: 1,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  questName: {
    fontSize: 15,
    color: '#fff',
  },
  questReward: {
    fontSize: 14,
    color: '#4CAF50',
    fontWeight: 'bold',
  },
  statsGrid: {
    flexDirection: 'row',
    gap: 12,
  },
  statCard: {
    flex: 1,
    backgroundColor: '#1a1a1a',
    borderRadius: 12,
    padding: 16,
    alignItems: 'center',
  },
  statValue: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#fff',
    marginTop: 8,
  },
  statLabel: {
    fontSize: 11,
    color: '#888',
    marginTop: 4,
    textAlign: 'center',
  },
  badgesGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 12,
  },
  badgeCard: {
    width: '30%',
    backgroundColor: '#1a1a1a',
    borderRadius: 12,
    padding: 16,
    alignItems: 'center',
  },
  badgeIcon: {
    fontSize: 32,
    marginBottom: 8,
  },
  badgeName: {
    fontSize: 11,
    color: '#fff',
    textAlign: 'center',
  },
  noBadges: {
    fontSize: 14,
    color: '#888',
    textAlign: 'center',
    padding: 24,
  },
  actionsContainer: {
    flexDirection: 'row',
    gap: 12,
    marginTop: 16,
  },
  actionButton: {
    flex: 1,
    backgroundColor: '#4CAF50',
    borderRadius: 12,
    padding: 16,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
  },
  actionButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
  },
});
