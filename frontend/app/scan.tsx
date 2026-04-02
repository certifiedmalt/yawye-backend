import { useState, useEffect } from 'react';
import {
  View,
  Text,
  TextInput,
  StyleSheet,
  TouchableOpacity,
  Alert,
  ActivityIndicator,
  Platform,
  Image,
} from 'react-native';
import { useRouter } from 'expo-router';
import { Camera, CameraView } from 'expo-camera';
import { BarcodeScanningResult } from 'expo-camera';
import { Ionicons } from '@expo/vector-icons';
import axios from 'axios';
import { useAuth } from '../context/AuthContext';

const BACKEND_URL = process.env.EXPO_PUBLIC_BACKEND_URL || 'https://web-production-66c05.up.railway.app';

export default function Scan() {
  const [hasPermission, setHasPermission] = useState<boolean | null>(null);
  const [scanned, setScanned] = useState(false);
  const [loading, setLoading] = useState(false);
  const [scannedBarcode, setScannedBarcode] = useState<string>('');
  const [showManualInput, setShowManualInput] = useState(false);
  const [manualBarcode, setManualBarcode] = useState('');
  const [confirmData, setConfirmData] = useState<any>(null);
  const [confirmBarcode, setConfirmBarcode] = useState<string>('');
  const [confirmNeedsAnalysis, setConfirmNeedsAnalysis] = useState(false);
  const { token } = useAuth();
  const router = useRouter();

  useEffect(() => {
    requestCameraPermission();
  }, []);

  const requestCameraPermission = async () => {
    try {
      const { status } = await Camera.requestCameraPermissionsAsync();
      console.log('Camera permission status:', status);
      setHasPermission(status === 'granted');
      
      if (status !== 'granted') {
        Alert.alert(
          'Camera Permission Required',
          'Please enable camera access in your device settings to scan barcodes.',
          [
            { text: 'Cancel', style: 'cancel' },
            { text: 'Try Again', onPress: requestCameraPermission }
          ]
        );
      }
    } catch (error) {
      console.error('Error requesting camera permission:', error);
      Alert.alert('Error', 'Failed to request camera permission');
      setHasPermission(false);
    }
  };

  const proceedWithProduct = async (quickData: any, barcode: string, needsAnalysis: boolean) => {
    // Update gamification streak (non-blocking)
    try {
      await axios.post(
        `${BACKEND_URL}/api/gamification/update-streak`,
        {},
        { headers: { Authorization: `Bearer ${token}` } }
      );
    } catch (e: any) {
      console.log('Gamification streak update failed', e?.response?.data || e?.message);
    }

    if (needsAnalysis) {
      router.push({
        pathname: '/result',
        params: { 
          productData: JSON.stringify(quickData),
          barcode: barcode,
          needsAnalysis: 'true'
        },
      });
    } else {
      router.push({
        pathname: '/result',
        params: { productData: JSON.stringify(quickData) },
      });
    }
  };

  const handleBarCodeScanned = async ({ data }: BarcodeScanningResult) => {
    if (scanned || loading) return;
    
    setScanned(true);
    setScannedBarcode(data);
    setLoading(true);

    try {
      console.log('Scanned barcode:', data);
      
      // Stage 1: Quick lookup - get product name/image fast
      const quickResponse = await axios.post(
        `${BACKEND_URL}/api/scan/quick`,
        { barcode: data },
        { headers: { Authorization: `Bearer ${token}` }, timeout: 15000 }
      );

      const quickData = quickResponse.data;
      const needsAnalysis = quickData.status !== 'complete';
      
      // Show confirmation screen with product info
      setConfirmData(quickData);
      setConfirmBarcode(data);
      setConfirmNeedsAnalysis(needsAnalysis);
      setLoading(false);

    } catch (error: any) {
      console.error('Scan error:', error.response?.data);
      
      // If quick scan returns 404, fall back to full scan
      if (error.response?.status === 404) {
        try {
          const fullResponse = await axios.post(
            `${BACKEND_URL}/api/scan`,
            { barcode: data },
            { headers: { Authorization: `Bearer ${token}` }, timeout: 45000 }
          );
          
          // Show confirmation for full scan result too
          setConfirmData(fullResponse.data);
          setConfirmBarcode(data);
          setConfirmNeedsAnalysis(false);
          setLoading(false);
          return;
        } catch (fallbackError: any) {
          console.error('Fallback scan also failed:', fallbackError.response?.data);
          const fallbackDetail = fallbackError.response?.data?.detail;
          const fallbackMessage = typeof fallbackDetail === 'object'
            ? `${fallbackDetail.message}${fallbackDetail.suggestion ? '\n\n' + fallbackDetail.suggestion : ''}`
            : fallbackDetail || 'Product not found in our database';
          
          Alert.alert(
            'Product Not Found',
            `Barcode: ${data}\n\n${fallbackMessage}`,
            [
              { text: 'Try Again', onPress: () => { setScanned(false); setScannedBarcode(''); setLoading(false); } },
              { text: 'Manual Entry', onPress: () => { setShowManualInput(true); setLoading(false); } },
              { text: 'Go Back', style: 'cancel', onPress: () => router.back() },
            ]
          );
          return;
        }
      }
      
      const detail = error.response?.data?.detail;
      const errorMessage = typeof detail === 'object' 
        ? `${detail.message}${detail.suggestion ? '\n\n' + detail.suggestion : ''}`
        : detail || 'Failed to scan product';
      
      Alert.alert(
        'Scan Failed',
        `Barcode: ${data}\n\n${errorMessage}`,
        [
          {
            text: 'Try Again',
            onPress: () => {
              setScanned(false);
              setScannedBarcode('');
              setLoading(false);
            },
          },
          {
            text: 'Manual Entry',
            onPress: () => {
              setShowManualInput(true);
              setLoading(false);
            },
          },
          {
            text: 'Go Back',
            style: 'cancel',
            onPress: () => router.back(),
          },
        ]
      );
    }
  };

  const handleManualEntry = async (manualCode: string) => {
    if (!manualCode.trim()) {
      Alert.alert('Error', 'Please enter a barcode');
      return;
    }
    
    setLoading(true);
    setScannedBarcode(manualCode);

    try {
      const quickResponse = await axios.post(
        `${BACKEND_URL}/api/scan/quick`,
        { barcode: manualCode },
        { headers: { Authorization: `Bearer ${token}` }, timeout: 15000 }
      );

      const quickData = quickResponse.data;
      const needsAnalysis = quickData.status !== 'complete';

      // Show confirmation screen
      setConfirmData(quickData);
      setConfirmBarcode(manualCode);
      setConfirmNeedsAnalysis(needsAnalysis);
      setShowManualInput(false);
      setLoading(false);
    } catch (error: any) {
      const detail = error.response?.data?.detail;
      const errorMessage = typeof detail === 'object'
        ? `${detail.message}${detail.suggestion ? '\n\n' + detail.suggestion : ''}`
        : detail || 'Product not found';
      Alert.alert(
        'Scan Failed',
        errorMessage,
        [
          {
            text: 'Try Again',
            onPress: () => {
              setShowManualInput(false);
              setScanned(false);
              setScannedBarcode('');
              setLoading(false);
            },
          },
        ]
      );
      setLoading(false);
    }
  };

  const resetScan = () => {
    setConfirmData(null);
    setConfirmBarcode('');
    setConfirmNeedsAnalysis(false);
    setScanned(false);
    setScannedBarcode('');
    setLoading(false);
  };

  // Product confirmation screen
  if (confirmData) {
    const productName = confirmData.product_name || 'Unknown Product';
    const brand = confirmData.brands || '';
    const imageUrl = confirmData.image_url;

    return (
      <View style={styles.container}>
        <View style={styles.confirmCard}>
          {imageUrl ? (
            <Image source={{ uri: imageUrl }} style={styles.confirmImage} />
          ) : (
            <View style={[styles.confirmImage, styles.confirmNoImage]}>
              <Ionicons name="cube-outline" size={48} color="#555" />
            </View>
          )}
          
          <Text style={styles.confirmTitle}>Is this your product?</Text>
          <Text style={styles.confirmName}>{productName}</Text>
          {brand ? <Text style={styles.confirmBrand}>{brand}</Text> : null}
          <Text style={styles.confirmBarcode}>Barcode: {confirmBarcode}</Text>

          <TouchableOpacity
            style={styles.confirmYes}
            onPress={() => proceedWithProduct(confirmData, confirmBarcode, confirmNeedsAnalysis)}
            data-testid="confirm-product-yes"
          >
            <Ionicons name="checkmark-circle" size={22} color="#fff" />
            <Text style={styles.confirmYesText}>Yes, analyse this</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={styles.confirmNo}
            onPress={resetScan}
            data-testid="confirm-product-no"
          >
            <Ionicons name="close-circle" size={22} color="#FF5252" />
            <Text style={styles.confirmNoText}>Not this product</Text>
          </TouchableOpacity>
        </View>
      </View>
    );
  }

  // Manual input UI
  if (showManualInput) {
    return (
      <View style={styles.container}>
        <Ionicons name="create-outline" size={80} color="#4CAF50" />
        <Text style={styles.text}>Manual Barcode Entry</Text>
        {scannedBarcode && (
          <Text style={[styles.text, { fontSize: 14, marginTop: 8, color: '#888' }]}>
            Scanned: {scannedBarcode}
          </Text>
        )}
        <Text style={[styles.text, { fontSize: 14, marginTop: 16 }]}>
          Correct the barcode if needed:
        </Text>
        <TextInput
          style={styles.input}
          placeholder="Enter barcode"
          placeholderTextColor="#666"
          value={manualBarcode}
          onChangeText={setManualBarcode}
          keyboardType="numeric"
          autoFocus
        />
        <TouchableOpacity 
          style={styles.button} 
          onPress={() => handleManualEntry(manualBarcode)}
          disabled={loading}
        >
          <Text style={styles.buttonText}>
            {loading ? 'Searching...' : 'Search Product'}
          </Text>
        </TouchableOpacity>
        <TouchableOpacity 
          style={[styles.button, { backgroundColor: '#333', marginTop: 16 }]} 
          onPress={() => {
            setShowManualInput(false);
            setScanned(false);
            setScannedBarcode('');
            setManualBarcode('');
          }}
        >
          <Text style={styles.buttonText}>Cancel</Text>
        </TouchableOpacity>
      </View>
    );
  }

  // Web fallback - camera not available in web preview
  if (Platform.OS === 'web') {
    return (
      <View style={styles.container}>
        <Ionicons name="barcode-outline" size={80} color="#4CAF50" />
        <Text style={styles.text}>Camera not available on web</Text>
        <Text style={[styles.text, { fontSize: 14, marginTop: 8 }]}>
          Use Expo Go app on your phone to scan barcodes
        </Text>
        <Text style={[styles.text, { fontSize: 14, marginTop: 16 }]}>
          Or enter a barcode manually for testing:
        </Text>
        <TextInput
          style={styles.input}
          placeholder="Enter barcode (e.g. 3017620422003)"
          placeholderTextColor="#666"
          value={manualBarcode}
          onChangeText={setManualBarcode}
          keyboardType="numeric"
        />
        <TouchableOpacity 
          style={styles.button} 
          onPress={() => handleManualEntry(manualBarcode)}
          disabled={loading}
        >
          <Text style={styles.buttonText}>
            {loading ? 'Scanning...' : 'Scan Manually'}
          </Text>
        </TouchableOpacity>
      </View>
    );
  }

  if (hasPermission === null) {
    return (
      <View style={styles.container}>
        <ActivityIndicator size="large" color="#4CAF50" />
        <Text style={styles.text}>Requesting camera permission...</Text>
      </View>
    );
  }

  if (hasPermission === false) {
    return (
      <View style={styles.container}>
        <Ionicons name="camera-off" size={64} color="#888" />
        <Text style={styles.text}>Camera permission denied</Text>
        <Text style={[styles.text, { fontSize: 14, marginTop: 8, color: '#888' }]}>
          Please enable camera access in your device settings
        </Text>
        <TouchableOpacity style={styles.button} onPress={requestCameraPermission}>
          <Text style={styles.buttonText}>Grant Permission</Text>
        </TouchableOpacity>
        <TouchableOpacity 
          style={[styles.button, { backgroundColor: '#333', marginTop: 16 }]} 
          onPress={() => router.back()}
        >
          <Text style={styles.buttonText}>Go Back</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <CameraView
        style={styles.camera}
        facing="back"
        onBarcodeScanned={scanned ? undefined : handleBarCodeScanned}
        barcodeScannerSettings={{
          barcodeTypes: [
            'ean13',
            'ean8',
            'upc_a',
            'upc_e',
            'code128',
            'code39',
            'qr',
          ],
        }}
      >
        <View style={styles.overlay}>
          <View style={styles.topOverlay}>
            <TouchableOpacity 
              style={styles.manualButton}
              onPress={() => setShowManualInput(true)}
            >
              <Ionicons name="create-outline" size={20} color="#fff" />
              <Text style={styles.manualButtonText}>Manual Entry</Text>
            </TouchableOpacity>
          </View>
          <View style={styles.middleRow}>
            <View style={styles.sideOverlay} />
            <View style={styles.scanArea}>
              <View style={[styles.corner, styles.cornerTL]} />
              <View style={[styles.corner, styles.cornerTR]} />
              <View style={[styles.corner, styles.cornerBL]} />
              <View style={[styles.corner, styles.cornerBR]} />
            </View>
            <View style={styles.sideOverlay} />
          </View>
          <View style={styles.bottomOverlay}>
            <Text style={styles.instructions}>
              {loading ? 'Analyzing product...' : 'Align barcode within the frame'}
            </Text>
            {scannedBarcode && !loading && (
              <Text style={styles.scannedCode}>Last scanned: {scannedBarcode}</Text>
            )}
            {loading && <ActivityIndicator size="large" color="#4CAF50" />}
          </View>
        </View>
      </CameraView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0c0c0c',
    justifyContent: 'center',
    alignItems: 'center',
  },
  camera: {
    flex: 1,
    width: '100%',
  },
  overlay: {
    flex: 1,
  },
  topOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.6)',
    paddingTop: 50,
    paddingHorizontal: 20,
    alignItems: 'flex-end',
  },
  manualButton: {
    flexDirection: 'row',
    backgroundColor: '#4CAF50',
    paddingHorizontal: 16,
    paddingVertical: 10,
    borderRadius: 20,
    alignItems: 'center',
  },
  manualButtonText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: 'bold',
    marginLeft: 6,
  },
  scannedCode: {
    color: '#4CAF50',
    fontSize: 14,
    marginTop: 12,
    fontWeight: '600',
  },
  middleRow: {
    flexDirection: 'row',
    height: 250,
  },
  sideOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.6)',
  },
  scanArea: {
    width: 300,
    position: 'relative',
  },
  corner: {
    position: 'absolute',
    width: 30,
    height: 30,
    borderColor: '#4CAF50',
  },
  cornerTL: {
    top: 0,
    left: 0,
    borderTopWidth: 4,
    borderLeftWidth: 4,
  },
  cornerTR: {
    top: 0,
    right: 0,
    borderTopWidth: 4,
    borderRightWidth: 4,
  },
  cornerBL: {
    bottom: 0,
    left: 0,
    borderBottomWidth: 4,
    borderLeftWidth: 4,
  },
  cornerBR: {
    bottom: 0,
    right: 0,
    borderBottomWidth: 4,
    borderRightWidth: 4,
  },
  bottomOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.6)',
    justifyContent: 'center',
    alignItems: 'center',
    paddingTop: 32,
  },
  instructions: {
    color: '#fff',
    fontSize: 16,
    textAlign: 'center',
    marginBottom: 16,
  },
  text: {
    color: '#fff',
    fontSize: 18,
    marginTop: 24,
  },
  input: {
    backgroundColor: '#1a1a1a',
    borderRadius: 12,
    padding: 16,
    color: '#fff',
    fontSize: 16,
    marginTop: 16,
    marginBottom: 16,
    borderWidth: 1,
    borderColor: '#333',
    width: '80%',
  },
  button: {
    backgroundColor: '#4CAF50',
    borderRadius: 12,
    padding: 16,
    marginTop: 24,
  },
  buttonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
  },
  confirmCard: {
    alignItems: 'center',
    backgroundColor: '#1a1a1a',
    borderRadius: 20,
    padding: 28,
    marginHorizontal: 24,
    width: '85%',
  },
  confirmImage: {
    width: 140,
    height: 140,
    borderRadius: 16,
    marginBottom: 20,
  },
  confirmNoImage: {
    backgroundColor: '#252525',
    justifyContent: 'center',
    alignItems: 'center',
  },
  confirmTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#fff',
    marginBottom: 12,
  },
  confirmName: {
    fontSize: 18,
    fontWeight: '600',
    color: '#4CAF50',
    textAlign: 'center',
    marginBottom: 4,
  },
  confirmBrand: {
    fontSize: 14,
    color: '#aaa',
    marginBottom: 8,
  },
  confirmBarcode: {
    fontSize: 12,
    color: '#666',
    marginBottom: 24,
  },
  confirmYes: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#4CAF50',
    borderRadius: 14,
    paddingVertical: 14,
    paddingHorizontal: 32,
    width: '100%',
    justifyContent: 'center',
    marginBottom: 12,
  },
  confirmYesText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
    marginLeft: 8,
  },
  confirmNo: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#252525',
    borderRadius: 14,
    paddingVertical: 14,
    paddingHorizontal: 32,
    width: '100%',
    justifyContent: 'center',
  },
  confirmNoText: {
    color: '#FF5252',
    fontSize: 16,
    fontWeight: '600',
    marginLeft: 8,
  },
});
