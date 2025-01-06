"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { useTheme } from "next-themes";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Loader2,
  BookOpen,
  FileText,
  Lightbulb,
  Moon,
  Sun,
} from "lucide-react";
import { CourseDatabase } from "./components/course-database";
import { RecommendationCard } from "./components/recommendation-card";

// Mock function to simulate backend API call
const getRecommendation = async (title, description) => {
  await new Promise((resolve) => setTimeout(resolve, 1500));
  const keywords = [
    "react",
    "javascript",
    "web development",
    "machine learning",
    "data science",
  ];
  const content = (title + " " + description).toLowerCase();
  const foundKeyword = keywords.find((keyword) => content.includes(keyword));
  return foundKeyword
    ? `Advanced ${foundKeyword.charAt(0).toUpperCase() + foundKeyword.slice(1)}`
    : "General Programming Course";
};

export default function CourseRecommendationDashboard() {
  const [title, setTitle] = useState("");
  const [description, setDescription] = useState("");
  const [recommendation, setRecommendation] = useState(null);
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!title.trim() || !description.trim()) {
      setError("Please enter both course title and description");
      return;
    }
    setIsLoading(true);
    setError("");
    setRecommendation(null);

    try {
      // Step 1: Preprocessing
      const preprocessedData = await getPreprocessedData(title, description);
      const { nama, deskripsi } = preprocessedData;

      // Step 2: LDA
      const ldaResult = await getLdaTopic(nama, deskripsi);
      const { SI_topics, MK_topics } = ldaResult;

      // Step 3: Similarity
      const similarityResult = await getDirectedSimilarity(
        SI_topics,
        MK_topics
      );
      const { similarity_AB, similarity_BA } = similarityResult;

      // Step 4: Recommendation
      const recommendationResult = await getRecommendation(
        similarity_AB,
        similarity_BA
      );
      console.log(recommendation)
      setRecommendation(recommendationResult);
    } catch (err) {
      setError("Failed to get a recommendation. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  const getPreprocessedData = async (title, description) => {
    const response = await fetch("/api/py/preprocessing", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        type: "SI",
        nama: [title],
        deskripsi: [description],
      }),
    });
    if (!response.ok) {
      throw new Error("Failed to preprocess data");
    }
    return response.json();
  };

  const getLdaTopic = async (nama, deskripsi) => {
    const response = await fetch("/api/py/lda_topic", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        num_topics: 1,
        nama,
        deskripsi,
        alpha: "symmetric",
        beta: 0.8,
        top_n_words: 20,
      }),
    });
    if (!response.ok) {
      throw new Error("Failed to get LDA topics");
    }
    return response.json();
  };

  const getDirectedSimilarity = async (SI_topics, MK_topics) => {
    const response = await fetch("/api/py/directed_similarity", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        SI_topics,
        MK_topics,
      }),
    });
    if (!response.ok) {
      throw new Error("Failed to get directed similarity");
    }
    return response.json();
  };

  const getRecommendation = async (similarity_AB, similarity_BA) => {
    const response = await fetch("/api/py/recommendation", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        similarity_AB,
        similarity_BA,
      }),
    });
    if (!response.ok) {
      throw new Error("Failed to get recommendation");
    }
    return response.json();
  };

  return (
    <div className="container mx-auto p-4 min-h-screen bg-gray-50 dark:bg-gray-900">
      <Card className="shadow-md mb-8">
        <CardHeader className="bg-gray-100 dark:bg-gray-800 flex justify-between items-center">
          <CardTitle className="text-2xl font-bold flex items-center text-gray-800 dark:text-gray-200">
            <BookOpen className="mr-2" />
            MBKM Course Recognition Recommendation System
          </CardTitle>
          <CardDescription className="text-gray-600 dark:text-gray-400">
            Manage courses and get recommendations
          </CardDescription>
        </CardHeader>
        <CardContent className="mt-6">
          <Tabs defaultValue="recommend" className="w-full">
            <TabsList className="grid w-full grid-cols-2">
              <TabsTrigger value="recommend">Get Recommendation</TabsTrigger>
              <TabsTrigger value="database">Course Database</TabsTrigger>
            </TabsList>
            <TabsContent value="database">
              <CourseDatabase />
            </TabsContent>
            <TabsContent value="recommend">
              <form onSubmit={handleSubmit} className="space-y-6">
                <div className="space-y-2">
                  <Label
                    htmlFor="title"
                    className="text-lg font-medium text-gray-700 dark:text-gray-300"
                  >
                    Course Title
                  </Label>
                  <Input
                    id="title"
                    placeholder="Enter course title"
                    value={title}
                    onChange={(e) => setTitle(e.target.value)}
                    className="w-full transition-all duration-200"
                  />
                </div>
                <div className="space-y-2">
                  <Label
                    htmlFor="description"
                    className="text-lg font-medium text-gray-700 dark:text-gray-300"
                  >
                    Course Description
                  </Label>
                  <Textarea
                    id="description"
                    placeholder="Enter course description"
                    value={description}
                    onChange={(e) => setDescription(e.target.value)}
                    rows={5}
                    className="w-full transition-all duration-200"
                  />
                </div>
                <Button
                  type="submit"
                  onClick={handleSubmit}
                  disabled={isLoading}
                  className="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 rounded-md transition-all duration-200"
                >
                  {isLoading ? (
                    <>
                      <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                      Analyzing...
                    </>
                  ) : (
                    <>
                      <FileText className="mr-2 h-5 w-5" />
                      Get Recommendation
                    </>
                  )}
                </Button>
              </form>
              {recommendation && (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.2 }}
                >
                  <RecommendationCard recommendation={recommendation} />
                </motion.div>
              )}
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>

      {error && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-4 p-4 bg-red-50 dark:bg-red-900 border border-red-200 dark:border-red-700 text-red-700 dark:text-red-200 rounded-md shadow-sm"
          role="alert"
        >
          {error}
        </motion.div>
      )}
    </div>
  );
}
