<template>
  <div id="app">
    <TopBar />
    <div class="main-content">
      <ModelManager />
      <InferenceControls />
      <HeatmapSettings />
      <StreamCanvas />
      <LogsPanel />
    </div>
  </div>
</template>

<script>
import TopBar from './components/TopBar.vue'
import ModelManager from './components/ModelManager.vue'
import InferenceControls from './components/InferenceControls.vue'
import HeatmapSettings from './components/HeatmapSettings.vue'
import StreamCanvas from './components/StreamCanvas.vue'
import LogsPanel from './components/LogsPanel.vue'

export default {
  components: {
    TopBar,
    ModelManager,
    InferenceControls,
    HeatmapSettings,
    StreamCanvas,
    LogsPanel
  }
}
</script>

<style>
#app {
  display: flex;
  flex-direction: column;
  height: 100vh;
}
.main-content {
  display: flex;
  flex-wrap: wrap;
  padding: 10px;
  gap: 10px;
}
</style>
