import React, { createContext, useState, useContext, useEffect, useRef } from 'react';
import AsyncStorage from '@react-native-async-storage/async-storage';
import axios, { AxiosError } from 'axios';
import { AppState, AppStateStatus } from 'react-native';

const BACKEND_URL = process.env.EXPO_PUBLIC_BACKEND_URL || 'https://web-production-66c05.up.railway.app';

// Debug logging
if (__DEV__) {
  console.log('[AUTH] Backend URL:', BACKEND_URL);
}

// Global axios defaults - timeout on ALL requests
axios.defaults.timeout = 15000;

interface User {
  id: string;
  email: string;
  name: string;
  subscription_tier: string;
  total_scans: number;
}

interface AuthContextType {
  user: User | null;
  token: string | null;
  login: (email: string, password: string) => Promise<void>;
  register: (email: string, password: string, name: string) => Promise<void>;
  logout: () => Promise<void>;
  refreshUser: () => Promise<void>;
  loading: boolean;
  backendUrl: string;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const appState = useRef(AppState.currentState);

  useEffect(() => {
    loadStoredAuth();

    // Handle app coming back to foreground — refresh user data
    const subscription = AppState.addEventListener('change', (nextAppState: AppStateStatus) => {
      if (appState.current.match(/inactive|background/) && nextAppState === 'active') {
        if (token) {
          fetchUser(token).catch(() => {});
        }
      }
      appState.current = nextAppState;
    });

    return () => subscription.remove();
  }, []);

  // Global 401 interceptor — auto-logout on expired tokens
  useEffect(() => {
    const interceptor = axios.interceptors.response.use(
      (response) => response,
      async (error: AxiosError) => {
        if (error.response?.status === 401 && token) {
          // Token expired or invalid - auto logout
          const url = error.config?.url || '';
          // Don't auto-logout on login/register attempts
          if (!url.includes('/auth/login') && !url.includes('/auth/register') && !url.includes('/auth/forgot')) {
            await logout();
          }
        }
        return Promise.reject(error);
      }
    );

    return () => {
      axios.interceptors.response.eject(interceptor);
    };
  }, [token]);

  const loadStoredAuth = async () => {
    try {
      const storedToken = await AsyncStorage.getItem('token');
      if (storedToken) {
        setToken(storedToken);
        await fetchUser(storedToken);
      }
    } catch (error) {
      // Token invalid or expired — clear it silently
      await AsyncStorage.removeItem('token');
    } finally {
      setLoading(false);
    }
  };

  const fetchUser = async (authToken: string) => {
    try {
      const response = await axios.get(`${BACKEND_URL}/api/auth/me`, {
        headers: { Authorization: `Bearer ${authToken}` },
      });
      setUser(response.data);
    } catch (error) {
      await logout();
    }
  };

  const login = async (email: string, password: string) => {
    try {
      const response = await axios.post(`${BACKEND_URL}/api/auth/login`, {
        email,
        password,
      });
      const { token: newToken, user: newUser } = response.data;
      await AsyncStorage.setItem('token', newToken);
      setToken(newToken);
      setUser(newUser);
    } catch (error: any) {
      if (error.response) {
        throw new Error(error.response.data?.detail || 'Invalid email or password');
      } else if (error.code === 'ECONNABORTED') {
        throw new Error('Connection timed out. Please check your internet and try again.');
      } else {
        throw new Error('Cannot connect to server. Please check your internet connection.');
      }
    }
  };

  const register = async (email: string, password: string, name: string) => {
    try {
      const response = await axios.post(`${BACKEND_URL}/api/auth/register`, {
        email,
        password,
        name,
      });
      const { token: newToken, user: newUser } = response.data;
      await AsyncStorage.setItem('token', newToken);
      setToken(newToken);
      setUser(newUser);
    } catch (error: any) {
      if (error.response) {
        throw new Error(error.response.data?.detail || 'Registration failed');
      } else if (error.code === 'ECONNABORTED') {
        throw new Error('Connection timed out. Please check your internet and try again.');
      } else {
        throw new Error('Cannot connect to server. Please check your internet connection.');
      }
    }
  };

  const logout = async () => {
    await AsyncStorage.removeItem('token');
    await AsyncStorage.removeItem('notifications_scheduled');
    setToken(null);
    setUser(null);
  };

  const refreshUser = async () => {
    if (token) {
      await fetchUser(token);
    }
  };

  return (
    <AuthContext.Provider value={{ user, token, login, register, logout, refreshUser, loading, backendUrl: BACKEND_URL }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}
