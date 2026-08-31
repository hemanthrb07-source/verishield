import { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { api } from '../services/api'
import { Upload, FileText, Image, Video, Loader2, Zap } from 'lucide-react'

interface Props {
  onStart: () => void
  onComplete: (result: any) => void
  onError: (msg: string) => void
  isProcessing: boolean
}

type VerifyMode = 'full' | 'document' | 'deepfake' | 'face'

export function UploadPanel({ onStart, onComplete, onError, isProcessing }: Props) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [referenceFile, setReferenceFile] = useState<File | null>(null)
  const [userId, setUserId] = useState('')
  const [mode, setMode] = useState<VerifyMode>('full')

  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      setSelectedFile(acceptedFiles[0])
    }
  }, [])

  const onRefDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      setReferenceFile(acceptedFiles[0])
    }
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.png', '.jpg', '.jpeg', '.tiff', '.bmp'],
      'video/*': ['.mp4', '.avi', '.mov'],
      'application/pdf': ['.pdf'],
    },
    maxFiles: 1,
    maxSize: 50 * 1024 * 1024,
  })

  const { getRootProps: getRefRootProps, getInputProps: getRefInputProps } = useDropzone({
    onDrop: onRefDrop,
    accept: { 'image/*': ['.png', '.jpg', '.jpeg'] },
    maxFiles: 1,
  })

  const handleVerify = async () => {
    if (!selectedFile) return
    onStart()

    try {
      let result
      switch (mode) {
        case 'full':
          result = await api.verifyFull(selectedFile, referenceFile || undefined, userId || undefined)
          break
        case 'document':
          result = await api.verifyDocument(selectedFile, userId || undefined)
          break
        case 'deepfake':
          result = await api.verifyDeepfake(selectedFile, userId || undefined)
          break
        case 'face':
          result = await api.verifyFace(selectedFile, referenceFile || undefined, userId || undefined)
          break
      }
      onComplete(result)
    } catch (err: any) {
      onError(err.message || 'Verification failed')
    }
  }

  const getFileIcon = (name: string) => {
    if (name.endsWith('.pdf')) return <FileText className="w-8 h-8 text-amber-400" />
    if (name.match(/\.(mp4|avi|mov)$/i)) return <Video className="w-8 h-8 text-purple-400" />
    return <Image className="w-8 h-8 text-blue-400" />
  }

  return (
    <div className="card">
      <h2 className="text-lg font-semibold text-white mb-6">Upload for Verification</h2>

      {/* Mode Selection */}
      <div className="flex gap-2 mb-6">
        {([
          { key: 'full' as VerifyMode, label: 'Full Pipeline' },
          { key: 'document' as VerifyMode, label: 'Document' },
          { key: 'deepfake' as VerifyMode, label: 'Deepfake' },
          { key: 'face' as VerifyMode, label: 'Face Match' },
        ]).map(({ key, label }) => (
          <button
            key={key}
            onClick={() => setMode(key)}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-all
              ${mode === key
                ? 'bg-brand-600 text-white shadow-lg shadow-brand-500/20'
                : 'bg-gray-800 text-gray-400 hover:bg-gray-700 hover:text-gray-200'
              }`}
          >
            {label}
          </button>
        ))}
      </div>

      {/* Main Upload Area */}
      <div
        {...getRootProps()}
        className={`relative border-2 border-dashed rounded-2xl p-8 text-center transition-all cursor-pointer
          ${isDragActive
            ? 'border-brand-500 bg-brand-500/10'
            : selectedFile
              ? 'border-emerald-500/50 bg-emerald-500/5'
              : 'border-gray-700 hover:border-gray-600 bg-gray-800/30'
          }`}
      >
        <input {...getInputProps()} />

        {selectedFile ? (
          <div className="flex items-center justify-center gap-4">
            {getFileIcon(selectedFile.name)}
            <div className="text-left">
              <p className="text-sm font-medium text-white">{selectedFile.name}</p>
              <p className="text-xs text-gray-400">
                {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
              </p>
            </div>
            <button
              onClick={(e) => { e.stopPropagation(); setSelectedFile(null) }}
              className="text-gray-500 hover:text-gray-300 text-xs ml-4"
            >
              Remove
            </button>
          </div>
        ) : (
          <>
            <Upload className="w-12 h-12 text-gray-500 mx-auto mb-3" />
            <p className="text-sm text-gray-300 mb-1">
              Drag & drop a file here, or click to select
            </p>
            <p className="text-xs text-gray-500">
              Supports images, videos, and PDFs (max 50MB)
            </p>
          </>
        )}
      </div>

      {/* Reference Image (for face matching) */}
      {(mode === 'full' || mode === 'face') && (
        <div
          {...getRefRootProps()}
          className="mt-4 border border-dashed border-gray-700 rounded-xl p-4 text-center cursor-pointer hover:border-gray-600 transition-all"
        >
          <input {...getRefInputProps()} />
          {referenceFile ? (
            <div className="flex items-center justify-center gap-3">
              <Image className="w-5 h-5 text-emerald-400" />
              <span className="text-sm text-gray-200">{referenceFile.name}</span>
              <span className="text-xs text-gray-500">(reference)</span>
              <button
                onClick={(e) => { e.stopPropagation(); setReferenceFile(null) }}
                className="text-gray-500 hover:text-gray-300 text-xs"
              >
                Remove
              </button>
            </div>
          ) : (
            <p className="text-xs text-gray-500">
              Optional: Add reference image for face matching
            </p>
          )}
        </div>
      )}

      {/* User ID */}
      <div className="mt-4">
        <input
          type="text"
          placeholder="User ID (optional)"
          value={userId}
          onChange={(e) => setUserId(e.target.value)}
          className="input text-sm"
        />
      </div>

      {/* Verify Button */}
      <button
        onClick={handleVerify}
        disabled={!selectedFile || isProcessing}
        className="btn-primary w-full mt-6 disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {isProcessing ? (
          <>
            <Loader2 className="w-5 h-5 mr-2 animate-spin" />
            Processing...
          </>
        ) : (
          <>
            <Zap className="w-5 h-5 mr-2" />
            Run Verification
          </>
        )}
      </button>
    </div>
  )
}
