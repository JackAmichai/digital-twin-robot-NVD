{{/*
Expand the name of the chart.
*/}}
{{- define "digital-twin.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "digital-twin.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "digital-twin.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "digital-twin.labels" -}}
helm.sh/chart: {{ include "digital-twin.chart" . }}
{{ include "digital-twin.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/part-of: digital-twin-robotics
{{- end }}

{{/*
Selector labels
*/}}
{{- define "digital-twin.selectorLabels" -}}
app.kubernetes.io/name: {{ include "digital-twin.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "digital-twin.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "digital-twin.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Return the proper image name for a component
*/}}
{{- define "digital-twin.image" -}}
{{- $registryName := .imageRoot.registry -}}
{{- $repositoryName := .imageRoot.repository -}}
{{- $tag := .imageRoot.tag | toString -}}
{{- if $registryName }}
{{- printf "%s/%s:%s" $registryName $repositoryName $tag -}}
{{- else }}
{{- printf "%s:%s" $repositoryName $tag -}}
{{- end }}
{{- end }}

{{/*
Return the proper Docker Image Registry Secret Names
*/}}
{{- define "digital-twin.imagePullSecrets" -}}
{{- if .Values.global.imagePullSecrets }}
imagePullSecrets:
{{- range .Values.global.imagePullSecrets }}
  - name: {{ . }}
{{- end }}
{{- end }}
{{- end }}

{{/*
GPU Resource Template
*/}}
{{- define "digital-twin.gpuResources" -}}
{{- if .Values.global.gpu.enabled }}
nvidia.com/gpu: {{ .gpuCount | default 1 | quote }}
{{- end }}
{{- end }}

{{/*
Common environment variables for NVIDIA services
*/}}
{{- define "digital-twin.nvidiaEnv" -}}
- name: NVIDIA_VISIBLE_DEVICES
  value: "all"
- name: NVIDIA_DRIVER_CAPABILITIES
  value: "all"
{{- end }}

{{/*
Redis connection string
*/}}
{{- define "digital-twin.redisHost" -}}
{{- if .Values.redis.enabled }}
{{- printf "%s-redis-master" .Release.Name }}
{{- else }}
{{- .Values.externalRedis.host }}
{{- end }}
{{- end }}

{{/*
Prometheus annotations for scraping
*/}}
{{- define "digital-twin.prometheusAnnotations" -}}
prometheus.io/scrape: "true"
prometheus.io/port: {{ .metricsPort | quote }}
prometheus.io/path: {{ .metricsPath | default "/metrics" | quote }}
{{- end }}
