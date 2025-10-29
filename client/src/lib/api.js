// API utility functions for client-side requests

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || '/api';

class ApiClient {
  constructor() {
    this.baseURL = API_BASE_URL;
  }

  async request(endpoint, options = {}) {
    const url = `${this.baseURL}${endpoint}`;
    
    const config = {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    };

    if (config.body && typeof config.body === 'object') {
      config.body = JSON.stringify(config.body);
    }

    try {
      const response = await fetch(url, config);
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || `HTTP error! status: ${response.status}`);
      }

      return data;
    } catch (error) {
      console.error('API request failed:', error);
      throw error;
    }
  }

  // User API methods
  async getCurrentUser() {
    return this.request('/users');
  }

  async updateUser(userData) {
    return this.request('/users', {
      method: 'PUT',
      body: userData,
    });
  }

  async completeProfile(profileData) {
    return this.request('/users/profile', {
      method: 'POST',
      body: profileData,
    });
  }

  async checkProfileStatus() {
    return this.request('/users/profile');
  }

  // Therapist API methods
  async getPatients() {
    return this.request('/therapists');
  }

  async getTherapistDashboard() {
    return this.request('/therapists', {
      method: 'POST',
    });
  }

  // Utility methods
  async healthCheck() {
    return this.request('/health');
  }
}

export const apiClient = new ApiClient();

// Hook for React components
export function useApi() {
  return apiClient;
}