# Secondary Speaker Removal Visualization

This is an interactive visualization of the Secondary Speaker Removal implementation plan for the Thai Audio Dataset Collection project.

## Features

- **Modern Glassmorphism Design**: Beautiful glass-like UI elements with blur effects
- **Interactive Components**: 
  - Tab navigation for technology stack
  - Animated timeline showing implementation phases
  - Progress bars with animation
  - Copy-to-clipboard for code blocks
- **Responsive Design**: Works seamlessly on desktop and mobile devices
- **Dark/Light Mode**: Toggle between themes
- **Smooth Animations**: Floating particles, wave animations, and scroll effects
- **Accessibility**: Keyboard navigation and screen reader support

## Viewing the Visualization

1. Open `index.html` in a modern web browser
2. Navigate through sections using the top navigation menu
3. Click on tabs to explore different aspects of the technology stack
4. Hover over diagram nodes to see tooltips
5. Click the copy button on code blocks to copy configuration examples

## Structure

- `index.html` - Main HTML file with embedded critical CSS
- `styles.css` - Additional styling and animations
- `script.js` - Interactive functionality and animations
- `README.md` - This documentation file

## Browser Compatibility

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## Technologies Used

- HTML5 with semantic markup
- CSS3 with advanced animations and glassmorphism effects
- Vanilla JavaScript for interactivity
- Font Awesome for icons
- Inter font family for typography

## Performance Optimizations

- Hardware-accelerated animations
- Lazy loading of animations with Intersection Observer
- Efficient event handling
- Minimal external dependencies

## Customization

The visualization uses CSS custom properties (variables) for easy theming:

```css
:root {
    --primary: #6366f1;
    --secondary: #ec4899;
    --accent: #06b6d4;
    --success: #10b981;
    --warning: #f59e0b;
    --danger: #ef4444;
}
```

Modify these values in the `index.html` file to change the color scheme.

## Accessibility Features

- Semantic HTML structure
- ARIA labels where appropriate
- Keyboard navigation support
- Focus indicators
- Reduced motion support
- High contrast mode support

## Future Enhancements

- Add more interactive demos
- Include audio samples for before/after comparison
- Real-time metrics visualization
- Integration with live processing dashboard

---

Created for the Thai Audio Dataset Collection project to visualize the implementation plan for removing secondary speakers from audio samples.