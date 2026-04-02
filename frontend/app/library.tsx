import { useState, useEffect, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  FlatList,
  TouchableOpacity,
  Image,
  ActivityIndicator,
  RefreshControl,
} from 'react-native';
import { useRouter } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import { SafeAreaView } from 'react-native-safe-area-context';
import axios from 'axios';
import { useAuth } from '../context/AuthContext';

const BACKEND_URL = process.env.EXPO_PUBLIC_BACKEND_URL || 'https://web-production-66c05.up.railway.app';

interface ScanItem {
  _id: string;
  barcode: string;
  product_name: string;
  brands?: string;
  image_url?: string;
  scanned_at: string;
  source?: string;
  analysis?: {
    overall_score: number;
    processing_category?: string;
    harmful_ingredients?: any[];
    beneficial_ingredients?: any[];
    carcinogens_found?: any[];
    shocking_facts?: any[];
    chemical_breakdown?: any[];
    healthier_alternatives?: any[];
    recommendation?: string;
    upf_score?: string;
  };
}

export default function Library() {
  const { token } = useAuth();
  const router = useRouter();
  const [scans, setScans] = useState<ScanItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  const fetchScans = useCallback(async () => {
    try {
      const res = await axios.get(`${BACKEND_URL}/api/scans/history`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      setScans(res.data.scans || []);
    } catch (e) {
      console.error('Failed to fetch scan history:', e);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, [token]);

  useEffect(() => {
    fetchScans();
  }, [fetchScans]);

  const onRefresh = () => {
    setRefreshing(true);
    fetchScans();
  };

  const getScoreColor = (score: number) => {
    if (score >= 7) return '#00E676';
    if (score >= 4) return '#FFD54F';
    return '#FF5252';
  };

  const formatDate = (dateStr: string) => {
    const d = new Date(dateStr);
    const now = new Date();
    const diffMs = now.getTime() - d.getTime();
    const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));
    if (diffDays === 0) return 'Today';
    if (diffDays === 1) return 'Yesterday';
    if (diffDays < 7) return `${diffDays} days ago`;
    return d.toLocaleDateString('en-GB', { day: 'numeric', month: 'short' });
  };

  const openResult = (scan: ScanItem) => {
    const productData = {
      product_name: scan.product_name,
      brands: scan.brands || '',
      ingredients_text: '',
      image_url: scan.image_url || '',
      analysis: scan.analysis,
    };
    router.push({
      pathname: '/result',
      params: { productData: JSON.stringify(productData) },
    });
  };

  const renderScanItem = ({ item }: { item: ScanItem }) => {
    const score = item.analysis?.overall_score;
    const scoreColor = score != null ? getScoreColor(score) : '#666';
    const harmfulCount = item.analysis?.harmful_ingredients?.length || 0;
    const category = item.analysis?.processing_category;
    const categoryColor = category
      ? category.toLowerCase().includes('ultra')
        ? '#FF5252'
        : category.toLowerCase().includes('processed') && !category.toLowerCase().includes('minimally')
        ? '#FFA726'
        : category.toLowerCase().includes('minimally')
        ? '#8BC34A'
        : '#00E676'
      : '#555';

    return (
      <TouchableOpacity
        style={styles.scanCard}
        onPress={() => openResult(item)}
        activeOpacity={0.7}
        data-testid={`scan-item-${item.barcode}`}
      >
        {item.image_url ? (
          <Image source={{ uri: item.image_url }} style={styles.productThumb} />
        ) : (
          <View style={[styles.productThumb, styles.noImage]}>
            <Ionicons name="cube-outline" size={28} color="#555" />
          </View>
        )}

        <View style={styles.scanInfo}>
          <Text style={styles.scanName} numberOfLines={1}>{item.product_name}</Text>
          {item.brands ? <Text style={styles.scanBrand} numberOfLines={1}>{item.brands}</Text> : null}
          <View style={styles.scanMeta}>
            <Text style={styles.scanDate}>{formatDate(item.scanned_at)}</Text>
            {category && (
              <View style={[styles.categoryBadge, { backgroundColor: categoryColor + '22', borderColor: categoryColor, borderWidth: 1 }]}>
                <Text style={[styles.categoryText, { color: categoryColor }]}>{category}</Text>
              </View>
            )}
          </View>
          {harmfulCount > 0 && (
            <Text style={styles.harmfulText}>{harmfulCount} harmful ingredient{harmfulCount > 1 ? 's' : ''}</Text>
          )}
        </View>

        {score != null ? (
          <View style={[styles.scoreCircle, { borderColor: scoreColor }]}>
            <Text style={[styles.scoreText, { color: scoreColor }]}>{score}</Text>
            <Text style={styles.scoreTen}>/10</Text>
          </View>
        ) : (
          <View style={[styles.scoreCircle, { borderColor: '#444' }]}>
            <Ionicons name="hourglass-outline" size={18} color="#666" />
          </View>
        )}
      </TouchableOpacity>
    );
  };

  return (
    <SafeAreaView style={styles.container} edges={['top']}>
      <View style={styles.header}>
        <TouchableOpacity onPress={() => router.back()} style={styles.backBtn} data-testid="library-back-btn">
          <Ionicons name="arrow-back" size={24} color="#fff" />
        </TouchableOpacity>
        <Text style={styles.title}>My Library</Text>
        <View style={styles.countBadge}>
          <Text style={styles.countText}>{scans.length}</Text>
        </View>
      </View>

      {loading ? (
        <View style={styles.centered}>
          <ActivityIndicator size="large" color="#4CAF50" />
        </View>
      ) : scans.length === 0 ? (
        <View style={styles.centered}>
          <Ionicons name="scan-outline" size={64} color="#333" />
          <Text style={styles.emptyTitle}>No scans yet</Text>
          <Text style={styles.emptyText}>Scan your first product to start building your library</Text>
          <TouchableOpacity style={styles.scanNowBtn} onPress={() => router.push('/scan')} data-testid="library-scan-now-btn">
            <Ionicons name="scan" size={20} color="#fff" />
            <Text style={styles.scanNowText}>Scan Now</Text>
          </TouchableOpacity>
        </View>
      ) : (
        <FlatList
          data={scans}
          keyExtractor={(item) => item._id}
          renderItem={renderScanItem}
          contentContainerStyle={styles.listContent}
          refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor="#4CAF50" />}
          showsVerticalScrollIndicator={false}
        />
      )}
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
    paddingHorizontal: 20,
    paddingVertical: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#1a1a1a',
  },
  backBtn: {
    marginRight: 12,
  },
  title: {
    fontSize: 22,
    fontWeight: 'bold',
    color: '#fff',
    flex: 1,
  },
  countBadge: {
    backgroundColor: '#4CAF50',
    borderRadius: 12,
    paddingHorizontal: 10,
    paddingVertical: 4,
  },
  countText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: 'bold',
  },
  listContent: {
    padding: 16,
  },
  scanCard: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#1a1a1a',
    borderRadius: 14,
    padding: 14,
    marginBottom: 10,
  },
  productThumb: {
    width: 56,
    height: 56,
    borderRadius: 10,
    marginRight: 14,
  },
  noImage: {
    backgroundColor: '#252525',
    justifyContent: 'center',
    alignItems: 'center',
  },
  scanInfo: {
    flex: 1,
    marginRight: 10,
  },
  scanName: {
    fontSize: 16,
    fontWeight: '600',
    color: '#fff',
    marginBottom: 2,
  },
  scanBrand: {
    fontSize: 13,
    color: '#888',
    marginBottom: 4,
  },
  scanMeta: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  scanDate: {
    fontSize: 12,
    color: '#666',
  },
  categoryBadge: {
    backgroundColor: '#252525',
    borderRadius: 6,
    paddingHorizontal: 6,
    paddingVertical: 2,
  },
  categoryText: {
    fontSize: 11,
    color: '#aaa',
  },
  harmfulText: {
    fontSize: 12,
    color: '#FF5252',
    marginTop: 2,
  },
  scoreCircle: {
    width: 48,
    height: 48,
    borderRadius: 24,
    borderWidth: 2.5,
    justifyContent: 'center',
    alignItems: 'center',
  },
  scoreText: {
    fontSize: 18,
    fontWeight: 'bold',
  },
  scoreTen: {
    fontSize: 10,
    color: '#666',
    marginTop: -2,
  },
  centered: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 40,
  },
  emptyTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#fff',
    marginTop: 16,
  },
  emptyText: {
    fontSize: 14,
    color: '#888',
    textAlign: 'center',
    marginTop: 8,
    marginBottom: 24,
  },
  scanNowBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#4CAF50',
    borderRadius: 24,
    paddingHorizontal: 24,
    paddingVertical: 12,
  },
  scanNowText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
    marginLeft: 8,
  },
});
