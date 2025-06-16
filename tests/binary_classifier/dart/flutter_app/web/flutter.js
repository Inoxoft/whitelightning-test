// Flutter Web Bootstrap Script
// This is a placeholder for the Flutter web loader
// The actual flutter.js will be generated during build

if (!window._flutter) {
  window._flutter = {};
}

window._flutter.loader = {
  loadEntrypoint: function(options) {
    console.log('Loading Flutter app...');
    if (options.onEntrypointLoaded) {
      // Simulate loading for development
      setTimeout(() => {
        options.onEntrypointLoaded({
          initializeEngine: () => Promise.resolve({
            runApp: () => console.log('Flutter app started')
          })
        });
      }, 100);
    }
  }
}; 