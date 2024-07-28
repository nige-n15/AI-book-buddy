package main

import (
	"bytes"
	"encoding/json"
	"github.com/gin-gonic/gin"
	"io/ioutil"
	"net/http"
)

type QueryRequest struct {
	Query string `json:"query"`
	TopK  int    `json:"top_k"`
}

type QueryResponse struct {
	Query             string          `json:"query"`
	AnthropicResponse string          `json:"anthropic_response"`
	RawResults        json.RawMessage `json:"raw_results"`
}

func main() {
	r := gin.Default()

	r.POST("/api/query", handleQuery)

	r.Run(":8080")
}

func handleQuery(c *gin.Context) {
	var req QueryRequest
	if err := c.BindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// Call Flask API
	flaskResp, err := callFlaskAPI(req)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Error calling Flask API"})
		return
	}

	var queryResp QueryResponse
	if err := json.Unmarshal(flaskResp, &queryResp); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Error parsing Flask API response"})
		return
	}

	c.JSON(http.StatusOK, queryResp)
}

func callFlaskAPI(req QueryRequest) ([]byte, error) {
	flaskURL := "http://localhost:5555/query"
	jsonData, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}

	resp, err := http.Post(flaskURL, "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	return body, nil
}
