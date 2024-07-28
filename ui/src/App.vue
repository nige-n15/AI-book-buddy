<script setup>
import { ref } from 'vue'
import axios from 'axios'

const query = ref('')
const topK = ref(5)
const response = ref(null)
const loading = ref(false)
const error = ref('')

const processQuery = async () => {
  loading.value = true
  error.value = ''
  response.value = null

  try {
    const result = await axios.post('http://localhost:8080/api/query', {
      query: query.value,
      top_k: topK.value
    })
    response.value = result.data
  } catch (err) {
    console.error('Error processing query:', err)
    error.value = 'An error occurred while processing the query.'
  } finally {
    loading.value = false
  }
}
</script>
<template>
  <div class="container">
    <h1>Query Processor</h1>
    <form @submit.prevent="processQuery">
      <div>
        <label for="query">Query:</label>
        <input id="query" v-model="query" required>
      </div>
      <div>
        <label for="topK">Top K:</label>
        <input id="topK" v-model.number="topK" type="number" required>
      </div>
      <button type="submit" :disabled="loading">Process Query</button>
    </form>
    <div v-if="loading" class="loading">Processing...</div>
    <div v-if="response" class="response">
      <h2>Response:</h2>
      <p><strong>Query:</strong> {{ response.query }}</p>
      <p><strong>Anthropic Response:</strong> {{ response.anthropic_response }}</p>
      <h3>Raw Results:</h3>
      <pre>{{ JSON.stringify(response.raw_results, null, 2) }}</pre>
    </div>
    <div v-if="error" class="error">
      {{ error }}
    </div>
  </div>
</template>


<style scoped>
.container {
  max-width: 800px;
  margin: 0 auto;
  padding: 20px;
}
form {
  display: flex;
  flex-direction: column;
  gap: 10px;
  margin-bottom: 20px;
}
input {
  width: 100%;
  padding: 5px;
}
button {
  align-self: flex-start;
  padding: 5px 10px;
}
.response {
  margin-top: 20px;
}
.loading, .error {
  margin-top: 20px;
  font-weight: bold;
}
.error {
  color: red;
}
pre {
  background-color: #f4f4f4;
  padding: 10px;
  border-radius: 5px;
  overflow-x: auto;
}
</style>
