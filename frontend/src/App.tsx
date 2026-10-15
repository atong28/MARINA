import { BrowserRouter, Routes, Route } from 'react-router-dom'
import MainPage from './pages/MainPage'
import RefinePage from './pages/RefinePage'

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<MainPage />} />
        <Route path="/refine/:id" element={<RefinePage />} />
      </Routes>
    </BrowserRouter>
  )
}

export default App
