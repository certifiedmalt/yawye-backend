import { View, Text, StyleSheet, ScrollView, TouchableOpacity, Linking } from 'react-native';
import { useRouter } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import { SafeAreaView } from 'react-native-safe-area-context';

export default function About() {
  const router = useRouter();

  return (
    <SafeAreaView style={styles.container} edges={['top']}>
      <View style={styles.header}>
        <TouchableOpacity onPress={() => router.back()} style={styles.backButton}>
          <Ionicons name="arrow-back" size={24} color="#fff" />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>About</Text>
        <View style={{ width: 24 }} />
      </View>

      <ScrollView contentContainerStyle={styles.content}>
        <View style={styles.logoSection}>
          <Ionicons name="scan-outline" size={60} color="#4CAF50" />
          <Text style={styles.appName}>You Are What You Eat</Text>
          <Text style={styles.tagline}>AI-Powered Ingredient Analysis</Text>
        </View>

        <View style={styles.section}>
          <View style={styles.badge}>
            <Ionicons name="shield-checkmark" size={20} color="#4CAF50" />
            <Text style={styles.badgeText}>Independent & Ad-Free</Text>
          </View>
        </View>

        <View style={styles.card}>
          <Text style={styles.cardTitle}>No Ads. No Agenda.</Text>
          <Text style={styles.cardText}>
            We're completely independent. No advertisers paying to promote their products. 
            No food companies influencing our scores. Just honest, science-based ingredient analysis.
          </Text>
        </View>

        <View style={styles.card}>
          <Text style={styles.cardTitle}>No Government Guidelines</Text>
          <Text style={styles.cardText}>
            We don't follow outdated government nutrition advice that tells you seed oils are 
            "heart healthy" or that ultra-processed cereals are part of a balanced breakfast. 
            Our AI analyzes based on independent research.
          </Text>
        </View>

        <View style={styles.card}>
          <Text style={styles.cardTitle}>Powered by AI + Open Data</Text>
          <Text style={styles.cardText}>
            We connect to the world's largest open food databases with millions of products. 
            Our AI analyzes ingredients in real-time - no waiting for products to be manually 
            added like other apps.
          </Text>
        </View>

        <View style={styles.card}>
          <Text style={styles.cardTitle}>What We Flag</Text>
          <View style={styles.bulletList}>
            <Text style={styles.bullet}>• Ultra-processed foods (UPF)</Text>
            <Text style={styles.bullet}>• Seed oils & industrial vegetable oils</Text>
            <Text style={styles.bullet}>• Artificial additives & preservatives</Text>
            <Text style={styles.bullet}>• Hidden sugars & inflammatory ingredients</Text>
            <Text style={styles.bullet}>• Chemical compounds the food industry hides</Text>
          </View>
        </View>

        <View style={styles.card}>
          <Text style={styles.cardTitle}>Our Mission</Text>
          <Text style={styles.cardText}>
            We believe you have the right to know exactly what's in your food - and WHY it matters. 
            Not just a score, but a real explanation of how ingredients affect your health.
          </Text>
          <Text style={[styles.cardText, styles.highlight]}>
            Stop trusting food labels. Start understanding ingredients.
          </Text>
        </View>

        <View style={styles.footer}>
          <Text style={styles.version}>Version 1.0.15</Text>
          <Text style={styles.copyright}>© 2026 You Are What You Eat</Text>
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
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#1a1a1a',
  },
  backButton: {
    padding: 4,
  },
  headerTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#fff',
  },
  content: {
    padding: 20,
  },
  logoSection: {
    alignItems: 'center',
    marginBottom: 24,
  },
  appName: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#fff',
    marginTop: 12,
  },
  tagline: {
    fontSize: 14,
    color: '#4CAF50',
    marginTop: 4,
  },
  section: {
    alignItems: 'center',
    marginBottom: 20,
  },
  badge: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(76, 175, 80, 0.15)',
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 20,
    borderWidth: 1,
    borderColor: 'rgba(76, 175, 80, 0.3)',
  },
  badgeText: {
    color: '#4CAF50',
    fontSize: 14,
    fontWeight: '600',
    marginLeft: 8,
  },
  card: {
    backgroundColor: '#1a1a1a',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
  },
  cardTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#fff',
    marginBottom: 8,
  },
  cardText: {
    fontSize: 14,
    color: '#aaa',
    lineHeight: 20,
  },
  highlight: {
    color: '#4CAF50',
    fontWeight: '600',
    marginTop: 12,
  },
  bulletList: {
    marginTop: 4,
  },
  bullet: {
    fontSize: 14,
    color: '#aaa',
    lineHeight: 24,
  },
  footer: {
    alignItems: 'center',
    marginTop: 20,
    paddingTop: 20,
    borderTopWidth: 1,
    borderTopColor: '#1a1a1a',
  },
  version: {
    fontSize: 12,
    color: '#666',
  },
  copyright: {
    fontSize: 12,
    color: '#444',
    marginTop: 4,
  },
});
