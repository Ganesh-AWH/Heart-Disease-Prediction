/*
  # Heart Disease Prediction Database Schema

  1. New Tables
    - `heart_disease_data`
      - `id` (uuid, primary key)
      - `age` (integer) - Patient age
      - `sex` (integer) - Sex (0=female, 1=male)
      - `cp` (integer) - Chest pain type (0-3)
      - `trestbps` (integer) - Resting blood pressure
      - `chol` (integer) - Serum cholesterol
      - `fbs` (integer) - Fasting blood sugar > 120 mg/dl
      - `restecg` (integer) - Resting ECG results
      - `thalach` (integer) - Maximum heart rate achieved
      - `exang` (integer) - Exercise induced angina
      - `oldpeak` (numeric) - ST depression
      - `slope` (integer) - Slope of peak exercise ST segment
      - `ca` (integer) - Number of major vessels
      - `thal` (integer) - Thalassemia
      - `target` (integer) - Heart disease presence (0=no, 1=yes)
      - `created_at` (timestamptz)

    - `model_training_results`
      - `id` (uuid, primary key)
      - `model_type` (text) - supervised or unsupervised
      - `model_name` (text) - Model algorithm name
      - `accuracy` (numeric) - Model accuracy (supervised only)
      - `precision` (numeric) - Model precision (supervised only)
      - `recall` (numeric) - Model recall (supervised only)
      - `f1_score` (numeric) - Model F1 score (supervised only)
      - `silhouette_score` (numeric) - Silhouette score (unsupervised only)
      - `davies_bouldin_score` (numeric) - DB index (unsupervised only)
      - `training_samples` (integer) - Number of samples used
      - `metadata` (jsonb) - Additional model metadata
      - `created_at` (timestamptz)

    - `predictions`
      - `id` (uuid, primary key)
      - `model_name` (text) - Model used for prediction
      - `input_data` (jsonb) - Patient input features
      - `prediction` (integer) - Predicted outcome
      - `probability` (jsonb) - Prediction probabilities
      - `cluster` (integer) - Cluster assignment (unsupervised)
      - `created_at` (timestamptz)

  2. Security
    - Enable RLS on all tables
    - Add policies for public access (for demonstration purposes)
    
  3. Important Notes
    - All tables use UUID primary keys
    - Timestamps are automatically set
    - JSONB used for flexible metadata storage
*/

-- Create heart_disease_data table
CREATE TABLE IF NOT EXISTS heart_disease_data (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  age integer NOT NULL,
  sex integer NOT NULL CHECK (sex IN (0, 1)),
  cp integer NOT NULL CHECK (cp BETWEEN 0 AND 3),
  trestbps integer NOT NULL,
  chol integer NOT NULL,
  fbs integer NOT NULL CHECK (fbs IN (0, 1)),
  restecg integer NOT NULL CHECK (restecg BETWEEN 0 AND 2),
  thalach integer NOT NULL,
  exang integer NOT NULL CHECK (exang IN (0, 1)),
  oldpeak numeric NOT NULL,
  slope integer NOT NULL CHECK (slope BETWEEN 0 AND 2),
  ca integer NOT NULL CHECK (ca BETWEEN 0 AND 4),
  thal integer NOT NULL CHECK (thal BETWEEN 0 AND 3),
  target integer NOT NULL CHECK (target IN (0, 1)),
  created_at timestamptz DEFAULT now()
);

-- Create model_training_results table
CREATE TABLE IF NOT EXISTS model_training_results (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  model_type text NOT NULL CHECK (model_type IN ('supervised', 'unsupervised')),
  model_name text NOT NULL,
  accuracy numeric,
  precision numeric,
  recall numeric,
  f1_score numeric,
  silhouette_score numeric,
  davies_bouldin_score numeric,
  training_samples integer NOT NULL DEFAULT 0,
  metadata jsonb DEFAULT '{}'::jsonb,
  created_at timestamptz DEFAULT now()
);

-- Create predictions table
CREATE TABLE IF NOT EXISTS predictions (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  model_name text NOT NULL,
  input_data jsonb NOT NULL,
  prediction integer,
  probability jsonb,
  cluster integer,
  created_at timestamptz DEFAULT now()
);

-- Enable Row Level Security
ALTER TABLE heart_disease_data ENABLE ROW LEVEL SECURITY;
ALTER TABLE model_training_results ENABLE ROW LEVEL SECURITY;
ALTER TABLE predictions ENABLE ROW LEVEL SECURITY;

-- Create policies for public access (for demonstration)
CREATE POLICY "Allow public read access to heart disease data"
  ON heart_disease_data
  FOR SELECT
  TO anon, authenticated
  USING (true);

CREATE POLICY "Allow public insert to heart disease data"
  ON heart_disease_data
  FOR INSERT
  TO anon, authenticated
  WITH CHECK (true);

CREATE POLICY "Allow public read access to training results"
  ON model_training_results
  FOR SELECT
  TO anon, authenticated
  USING (true);

CREATE POLICY "Allow public insert to training results"
  ON model_training_results
  FOR INSERT
  TO anon, authenticated
  WITH CHECK (true);

CREATE POLICY "Allow public read access to predictions"
  ON predictions
  FOR SELECT
  TO anon, authenticated
  USING (true);

CREATE POLICY "Allow public insert to predictions"
  ON predictions
  FOR INSERT
  TO anon, authenticated
  WITH CHECK (true);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_heart_disease_target ON heart_disease_data(target);
CREATE INDEX IF NOT EXISTS idx_model_results_type ON model_training_results(model_type);
CREATE INDEX IF NOT EXISTS idx_model_results_name ON model_training_results(model_name);
CREATE INDEX IF NOT EXISTS idx_predictions_model ON predictions(model_name);
CREATE INDEX IF NOT EXISTS idx_created_at ON heart_disease_data(created_at);
