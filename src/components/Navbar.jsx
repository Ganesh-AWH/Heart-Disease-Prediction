import { Link, useLocation } from 'react-router-dom';

export default function Navbar() {
  const location = useLocation();

  return (
    <nav className="navbar">
      <div className="navbar-container">
        <Link to="/" className="navbar-brand">
          <span className="navbar-icon">❤️</span>
          Heart Disease Prediction - PDS
        </Link>
        <ul className="navbar-nav">
          <li>
            <Link
              to="/"
              className={`nav-link ${location.pathname === '/' ? 'active' : ''}`}
            >
              Dashboard
            </Link>
          </li>
          <li>
            <Link
              to="/upload"
              className={`nav-link ${location.pathname === '/upload' ? 'active' : ''}`}
            >
              Upload Data
            </Link>
          </li>
          <li>
            <Link
              to="/train"
              className={`nav-link ${location.pathname === '/train' ? 'active' : ''}`}
            >
              Train Models
            </Link>
          </li>
          <li>
            <Link
              to="/predict"
              className={`nav-link ${location.pathname === '/predict' ? 'active' : ''}`}
            >
              Prediction
            </Link>
          </li>
          <li>
            <Link
              to="/comparison"
              className={`nav-link ${location.pathname === '/comparison' ? 'active' : ''}`}
            >
              Comparison
            </Link>
          </li>
        </ul>
      </div>
    </nav>
  );
}
