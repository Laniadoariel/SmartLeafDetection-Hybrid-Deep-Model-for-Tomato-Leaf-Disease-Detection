import axios from 'axios'

const api = axios.create({ baseURL: '/api' })

api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token')
  if (token) config.headers.Authorization = `Bearer ${token}`
  return config
})

api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401 && !window.location.pathname.includes('/login')) {
      localStorage.removeItem('token')
      localStorage.removeItem('username')
      window.location.href = '/login'
    }
    return Promise.reject(error)
  }
)

export default api

export interface FlightSummary {
  id: string; video_filename: string; status: string;
  current_stage: string; progress: number;
  total_frames: number; processed_frames: number;
  total_video_frames: number; relevant_frames: number; total_detections: number;
  total_plants: number; diseased_plants: number; healthy_plants: number;
  created_at: string; completed_at: string | null;
}

export interface LeafResult {
  leaf_id: number; frame_index: number; label: string; confidence: number;
  bbox: number[]; crop_path: string | null;
}

export interface PlantResult {
  id: string; plant_id: number; status: string;
  disease_labels: string[]; confidence: number;
  leaf_count: number; diseased_leaf_count: number;
  frames_seen: number; views_total: number; views_agreeing: number;
  weighted_decision: boolean;
  gps_lat: number | null; gps_lon: number | null;
  leaves: LeafResult[];
}

export interface FrameData {
  frame_index: number; original_path: string;
  annotated_path: string | null; plant_count: number; leaf_count: number;
}

export interface FlightDetail extends FlightSummary {
  error_message: string | null;
  plants: PlantResult[];
  frames: FrameData[];
}
