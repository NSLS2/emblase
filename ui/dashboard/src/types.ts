// Shared types for the EMBLASE Dashboard

export type ServiceStatus = 'online' | 'offline' | 'error' | 'timeout' | 'unconfigured' | 'placeholder' | 'degraded' | 'loading'

export interface TiledStatus {
  status: ServiceStatus
  server_uri?: string
  tiled_version?: string
  python_version?: string
  api_version?: string
  input_container?: string
  output_container?: string
  error?: string
  message?: string
}

export interface OrionStatus {
  status: ServiceStatus
  api_url?: string
  cluster?: string
  account?: string
  active_jobs?: number
  latest_job?: Record<string, unknown> | null
  error?: string
  message?: string
}

export interface NERSCStatus {
  status: ServiceStatus
  api_uri?: string
  resource_id?: string
  account?: string
  queue?: string
  container_image?: string
  time_limit?: string
  available_resources?: string[]
  error?: string
  message?: string
}

export interface MLflowStatus {
  status: ServiceStatus
  tracking_uri?: string
  experiment?: string
  model_prefix?: string
  error?: string
  message?: string
}

export interface MLflowModel {
  name: string
  latest_version: number | string | null
  description: string | null
}

export interface ChatbotStatus {
  status: ServiceStatus
  url?: string
  model?: string
  http_status?: number
  error?: string
  message?: string
}

export interface OpenMetadataStatus {
  status: 'placeholder'
  message: string
  planned_features: string[]
}

export interface AppConfig {
  tiled_server_uri: string
  tiled_input_container: string
  tiled_output_container: string
  mlflow_tracking_uri: string
  mlflow_experiment: string
  mlflow_model_prefix: string
  chatapp_url: string
  chatapp_model: string
  orion_api_url: string
  orion_cluster: string
  orion_account: string
  nersc_api_uri: string
  nersc_resource_id: string
  nersc_account: string
  nersc_queue: string
  nersc_time_limit: string
  nersc_container_image: string
  compute_backend: string
}

export interface JobSubmitResult {
  job_id: string
  backend: string
  log_path: string
  submitted_at: number
}

export interface ParamSpec {
  name: string
  source: string
  dtype: string
  units: string
}

export interface BatchJobRequest {
  backend: 'orion' | 'nersc'
  model_name: string
  mlflow_version: string
  input_container: string
  output_container: string
  batch_size: number
  image_key: string
  thumb_mode: string
  param_specs: ParamSpec[]
  projector?: string
  classifier?: string
  nersc_queue?: string
  nersc_account?: string
  nersc_time_limit?: string
  nersc_constraint?: string
  orion_account?: string
}

export interface StreamJobRequest {
  model_name: string
  mlflow_version: string
  run_path: string
  output_container: string
  batch_size: number
  image_key: string
  thumb_mode: string
  param_specs: ParamSpec[]
  projector?: string
  classifier?: string
  nersc_queue?: string
  nersc_account?: string
  nersc_time_limit?: string
  nersc_constraint?: string
}
