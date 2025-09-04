// src/main.ts
import './style.css';
import { CrownMuralController } from './CrownMuralController';

// Initialize when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => {
    new CrownMuralController();
  });
} else {
  new CrownMuralController();
}
