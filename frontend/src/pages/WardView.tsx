import React, { useMemo } from 'react'
import { usePatientStore, useUIStore } from '../stores'
import { PatientCard, getCategoryPriority } from '../components'
import type { Category } from '../types'

type SortMode = 'id' | 'category' | 'fhr' | 'time'
type FilterMode = 'all' | Category

export const WardView: React.FC = () => {
  const patients = usePatientStore(state => state.patients)
  const liveUpdates = usePatientStore(state => state.liveUpdates)
  const gridColumns = useUIStore(state => state.gridColumns)
  const setGridColumns = useUIStore(state => state.setGridColumns)
  
  const [sortMode, setSortMode] = React.useState<SortMode>('category')
  const [filterMode, setFilterMode] = React.useState<FilterMode>('all')
  const [searchQuery, setSearchQuery] = React.useState('')
  
  // Convert Map to array and apply sorting/filtering
  const sortedPatients = useMemo(() => {
    let patientArray = Array.from(patients.values())
    
    // Apply filter
    if (filterMode !== 'all') {
      patientArray = patientArray.filter(p => p.category === filterMode)
    }
    
    // Apply search
    if (searchQuery) {
      const query = searchQuery.toLowerCase()
      patientArray = patientArray.filter(p => 
        p.patient_id.toLowerCase().includes(query)
      )
    }
    
    // Apply sort
    switch (sortMode) {
      case 'category':
        patientArray.sort((a, b) => {
          const priorityDiff = getCategoryPriority(b.category) - 
                              getCategoryPriority(a.category)
          if (priorityDiff !== 0) return priorityDiff
          return a.patient_id.localeCompare(b.patient_id)
        })
        break
      case 'fhr':
        patientArray.sort((a, b) => (b.metrics.current_fhr ?? 0) - (a.metrics.current_fhr ?? 0))
        break
      case 'time':
        patientArray.sort((a, b) => 
          b.last_update - a.last_update
        )
        break
      case 'id':
      default:
        patientArray.sort((a, b) => a.patient_id.localeCompare(b.patient_id))
    }
    
    return patientArray
  }, [patients, sortMode, filterMode, searchQuery])
  
  // Category counts for filter badges
  const categoryCounts = useMemo(() => {
    const counts: Record<string, number> = { all: patients.size }
    for (const patient of patients.values()) {
      const cat = patient.category
      counts[cat] = (counts[cat] ?? 0) + 1
    }
    return counts
  }, [patients])
  
  const gridClass: Record<number, string> = {
    2: 'grid-cols-1 md:grid-cols-2',
    3: 'grid-cols-1 md:grid-cols-2 lg:grid-cols-3',
    4: 'grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4',
    6: 'grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-6'
  }
  
  return (
    <div className="p-6">
      {/* Toolbar */}
      <div className="mb-6 flex flex-wrap items-center gap-4">
        {/* Search */}
        <div className="relative flex-1 min-w-48 max-w-xs">
          <input
            type="text"
            placeholder="Search patients..."
            value={searchQuery}
            onChange={e => setSearchQuery(e.target.value)}
            className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-sm text-white placeholder-gray-500 focus:outline-none focus:border-blue-500"
          />
          {searchQuery && (
            <button 
              onClick={() => setSearchQuery('')}
              className="absolute right-2 top-1/2 -translate-y-1/2 text-gray-500 hover:text-white"
            >
              ✕
            </button>
          )}
        </div>
        
        {/* Filter buttons */}
        <div className="flex items-center gap-1 bg-gray-800 rounded-lg p-1">
          <FilterButton 
            active={filterMode === 'all'} 
            onClick={() => setFilterMode('all')}
            count={categoryCounts.all}
          >
            All
          </FilterButton>
          <FilterButton 
            active={filterMode === 3} 
            onClick={() => setFilterMode(3)}
            count={categoryCounts[3] ?? 0}
            color="red"
          >
            Critical
          </FilterButton>
          <FilterButton 
            active={filterMode === 2} 
            onClick={() => setFilterMode(2)}
            count={categoryCounts[2] ?? 0}
            color="yellow"
          >
            Warning
          </FilterButton>
          <FilterButton 
            active={filterMode === 1} 
            onClick={() => setFilterMode(1)}
            count={categoryCounts[1] ?? 0}
            color="green"
          >
            Normal
          </FilterButton>
        </div>
        
        {/* Sort dropdown */}
        <select
          value={sortMode}
          onChange={e => setSortMode(e.target.value as SortMode)}
          className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500"
        >
          <option value="category">Sort by Priority</option>
          <option value="id">Sort by ID</option>
          <option value="fhr">Sort by FHR</option>
          <option value="time">Sort by Time</option>
        </select>
        
        {/* Grid size */}
        <div className="flex items-center gap-2">
          <span className="text-xs text-gray-500">Grid:</span>
          {[2, 3, 4, 6].map(cols => (
            <button
              key={cols}
              onClick={() => setGridColumns(cols)}
              className={`w-8 h-8 rounded text-xs font-medium transition-colors ${
                gridColumns === cols 
                  ? 'bg-blue-600 text-white' 
                  : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
              }`}
            >
              {cols}
            </button>
          ))}
        </div>
      </div>
      
      {/* Patient Grid */}
      {sortedPatients.length > 0 ? (
        <div className={`grid ${gridClass[gridColumns] ?? 'grid-cols-3'} gap-4`}>
          {sortedPatients.map(patient => {
            // Get FHR history from live updates if available
            const liveUpdate = liveUpdates.get(patient.patient_id)
            const fhrHistory = liveUpdate?.fhr_latest ?? []
            
            return (
              <PatientCard 
                key={patient.patient_id} 
                patient={patient}
                compact={gridColumns >= 4}
                fhrHistory={fhrHistory}
              />
            )
          })}
        </div>
      ) : (
        <EmptyState 
          hasFilter={filterMode !== 'all' || !!searchQuery}
          onClear={() => { setFilterMode('all'); setSearchQuery('') }}
        />
      )}
    </div>
  )
}

// Filter button component
interface FilterButtonProps {
  active: boolean
  onClick: () => void
  count: number
  color?: 'red' | 'yellow' | 'green'
  children: React.ReactNode
}

const FilterButton: React.FC<FilterButtonProps> = ({ 
  active, 
  onClick, 
  count, 
  color,
  children 
}) => {
  const colorClasses = {
    red: active ? 'bg-red-600 text-white' : 'text-red-400 hover:bg-red-900/30',
    yellow: active ? 'bg-yellow-600 text-white' : 'text-yellow-400 hover:bg-yellow-900/30',
    green: active ? 'bg-green-600 text-white' : 'text-green-400 hover:bg-green-900/30'
  }
  
  const baseClass = active ? 'bg-gray-700 text-white' : 'text-gray-400 hover:bg-gray-700'
  const colorClass = color ? colorClasses[color] : baseClass
  
  return (
    <button
      onClick={onClick}
      className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${colorClass}`}
    >
      {children}
      {count > 0 && (
        <span className="ml-1.5 px-1.5 py-0.5 rounded bg-black/20 text-xs">
          {count}
        </span>
      )}
    </button>
  )
}

// Empty state component
const EmptyState: React.FC<{ hasFilter: boolean; onClear: () => void }> = ({ 
  hasFilter, 
  onClear 
}) => (
  <div className="flex flex-col items-center justify-center py-20 text-gray-500">
    <div className="text-6xl mb-4">🏥</div>
    <h3 className="text-xl font-medium mb-2">No patients found</h3>
    {hasFilter ? (
      <>
        <p className="text-sm mb-4">No patients match your current filters</p>
        <button
          onClick={onClear}
          className="px-4 py-2 bg-gray-800 hover:bg-gray-700 rounded-lg text-sm transition-colors"
        >
          Clear filters
        </button>
      </>
    ) : (
      <p className="text-sm">Start the simulation to see patient data</p>
    )}
  </div>
)

export default WardView
