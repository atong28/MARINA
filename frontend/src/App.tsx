import { BrowserRouter, Routes, Route } from 'react-router-dom'
import ModelBootstrap from './components/common/ModelBootstrap'
import MainPage from './pages/MainPage'

function App() {
  return (
    <BrowserRouter>
      <ModelBootstrap />
      <Routes>
        <Route path="/" element={<MainPage />} />
      </Routes>
    </BrowserRouter>
  )
}

export default App
