import { useEffect } from 'react'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { Toaster } from 'react-hot-toast'

// Initialize i18n before any component renders
import './i18n'

import { Layout } from './components/layout'
import { WardView, DetailView } from './pages'
import { usePatientStream } from './hooks'

// Create React Query client with sensible defaults
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5000,
      refetchOnWindowFocus: false,
      retry: 1
    }
  }
})

// WebSocket connection wrapper component
function WebSocketProvider({ children }: { children: React.ReactNode }) {
  const { connect, disconnect } = usePatientStream()
  
  useEffect(() => {
    // Connect to WebSocket when app mounts
    connect()
    
    // Disconnect when app unmounts
    return () => {
      disconnect()
    }
  }, [connect, disconnect])
  
  return <>{children}</>
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <WebSocketProvider>
          <Routes>
            <Route path="/" element={<Layout />}>
              <Route index element={<WardView />} />
              <Route path="patient/:patientId" element={<DetailView />} />
            </Route>
          </Routes>
          
          {/* Toast notifications */}
          <Toaster
            position="top-right"
            toastOptions={{
              duration: 4000,
              style: {
                background: '#ffffff',
                color: '#1a1a1a',
                border: '1px solid #e5e7eb',
                boxShadow: '0 4px 12px rgba(0, 0, 0, 0.1)',
              },
              success: {
                iconTheme: {
                  primary: '#22c55e',
                  secondary: '#ffffff',
                },
              },
              error: {
                iconTheme: {
                  primary: '#ef4444',
                  secondary: '#ffffff',
                },
              },
            }}
          />
        </WebSocketProvider>
      </BrowserRouter>
    </QueryClientProvider>
  )
}

export default App
