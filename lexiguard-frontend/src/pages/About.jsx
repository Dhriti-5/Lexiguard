import React from "react";
import { motion } from "framer-motion";
import { Shield, Brain, Award, Linkedin } from "lucide-react";
import krishaImg from "./krisha.jpg";
import dhritiImg from "./dhriti_pic.jpg";
import kavyaImg from "./kavya_pic.jpg";

const teamMembers = [
  {
    name: "Krisha Gandhi",
    role: "B.Tech Pre-Final Year, Computer Science & Engineering",
    bio: "Student at The Maharaja Sayajirao University of Baroda",
    image: krishaImg, // local image import
    linkedin: "https://www.linkedin.com/in/krisha-gandhi-2a35972aa",
    github: "https://github.com/krishagandhi0711",
  },
  {
    name: "Dhriti Gandhi",
    role: "B.Tech Pre-Final Year, Computer Science & Engineering",
    bio: "Student at The Maharaja Sayajirao University of Baroda",
    image: dhritiImg, // local image import",
    linkedin: "http://www.linkedin.com/in/dhriti-gandhi-0758372b5",
    github: "https://github.com/Dhriti-5",
  },
  {
    name: "Kavya Patel",
    role: "B.Tech Pre-Final Year, Computer Science & Engineering",
    bio: "Student at The Maharaja Sayajirao University of Baroda",
    image: kavyaImg, // local image import
    linkedin: "https://www.linkedin.com/in/kavya-patel-802959278",
    github: "https://github.com/K9Patel",
  },

];

const milestones = [
  {
    title: "The Idea",
    content:
      "We noticed that legal documents are complex and time-consuming to analyze. The thought was to create an AI-driven tool that makes legal analysis fast, accurate, and accessible.",
  },
  {
    title: "Implementation",
    content:
      "Our team combined expertise in AI, NLP, and legal knowledge to design algorithms that can detect risks, extract clauses, and score documents automatically.",
  },
  {
    title: "Product Launch",
    content:
      "After rigorous testing and iterations, we launched a product that empowers businesses and legal professionals to understand contracts in seconds rather than hours.",
  },
];

export default function AboutPage() {
  return (
    <div className="min-h-screen bg-gray-50 dark:bg-[#121212]">
      {/* Hero Section */}
      <section className="py-24">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="text-center"
          >
            <h1 className="text-4xl md:text-6xl font-bold text-gray-900 dark:text-[#EAEAEA] mb-6">
              Revolutionizing <span className="gradient-text">Legal Analysis</span>
            </h1>
            <p className="text-xl text-gray-600 dark:text-[#A0A0A0] max-w-3xl mx-auto">
              We're on a mission to make legal document analysis faster, more accurate,
              and accessible to everyone through the power of artificial intelligence.
            </p>
          </motion.div>
        </div>
      </section>

      {/* Mission Section */}
      <section className="py-20 bg-white dark:bg-[#1E1E1E]">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid lg:grid-cols-2 gap-16 items-center">
            <motion.div
              initial={{ opacity: 0, x: -30 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.8 }}
            >
              <h2 className="text-3xl md:text-4xl font-bold text-gray-900 dark:text-[#EAEAEA] mb-6">
                Our Mission
              </h2>
              <p className="text-lg text-gray-600 dark:text-[#A0A0A0] leading-relaxed mb-8">
                Legal documents shouldn't be a black box. We believe everyone deserves
                to understand the risks and implications of their legal agreements.
                By combining advanced AI with deep legal expertise, we're making
                professional-grade document analysis accessible to businesses of all sizes.
              </p>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
                <div className="flex items-start gap-3">
                  <div className="w-10 h-10 bg-gradient-to-r from-[#1C3D5A] to-[#14B8A6] rounded-lg flex items-center justify-center">
                    <Shield className="w-5 h-5 text-white" />
                  </div>
                  <div>
                    <h3 className="font-semibold text-gray-900 dark:text-[#EAEAEA] mb-1">
                      Security First
                    </h3>
                    <p className="text-sm text-gray-600 dark:text-[#A0A0A0]">
                      Enterprise-grade security for your sensitive documents
                    </p>
                  </div>
                </div>

                <div className="flex items-start gap-3">
                  <div className="w-10 h-10 bg-gradient-to-r from-[#1C3D5A] to-[#14B8A6] rounded-lg flex items-center justify-center">
                    <Brain className="w-5 h-5 text-white" />
                  </div>
                  <div>
                    <h3 className="font-semibold text-gray-900 dark:text-[#EAEAEA] mb-1">
                      AI-Powered
                    </h3>
                    <p className="text-sm text-gray-600 dark:text-[#A0A0A0]">
                      AI trained on millions of legal documents
                    </p>
                  </div>
                </div>
              </div>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, x: 30 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.8 }}
              className="relative"
            >
              <div className="aspect-[4/3] bg-gradient-to-br from-[#1C3D5A] to-[#14B8A6] rounded-2xl p-8 text-white flex flex-col justify-center items-center text-center">
                <Award className="w-16 h-16 mb-6" />
                <h3 className="text-2xl font-bold mb-4">99.8% Accuracy</h3>
                <p className="text-lg opacity-90">
                  Our AI achieves industry-leading accuracy in legal document analysis
                </p>
              </div>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Team Section */}
      <section className="py-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
            className="text-center mb-16"
          >
            <h2 className="text-3xl md:text-4xl font-bold text-gray-900 dark:text-[#EAEAEA] mb-4">
              Meet Our <span className="gradient-text">Team</span>
            </h2>
            <p className="text-lg text-gray-600 dark:text-[#A0A0A0] max-w-3xl mx-auto">
              Led by students from CSE at The Maharaja Sayajirao University of Baroda,
              passionate about building the future of AI-driven legal analysis.
            </p>
          </motion.div>

          <div className="flex flex-wrap justify-center gap-8">
            {teamMembers.map((member, index) => (
              <motion.div
                key={member.name}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                whileHover={{ y: -8, transition: { duration: 0.2 } }}
                className="w-80 bg-white dark:bg-[#1E1E1E] rounded-2xl p-8 text-center shadow-lg hover:shadow-xl transition-all duration-300"
              >
                <img
                  src={member.image}
                  alt={member.name}
                  className="w-24 h-24 rounded-full mx-auto mb-6 object-cover"
                />
                <h3 className="text-xl font-bold text-gray-900 dark:text-[#EAEAEA] mb-2">
                  {member.name}
                </h3>
                <p className="text-[#1C3D5A] dark:text-[#14B8A6] font-semibold mb-4">
                  {member.role}
                </p>
                <p className="text-gray-600 dark:text-[#A0A0A0] mb-6">{member.bio}</p>
                <div className="flex justify-center gap-4">
                  {/* LinkedIn */}
                  {member.linkedin && (
                    <a
                      href={member.linkedin}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="w-8 h-8 bg-gray-100 dark:bg-gray-800 rounded-full flex items-center justify-center hover:bg-[#1C3D5A] hover:text-white transition-colors"
                    >
                      <Linkedin className="w-4 h-4" />
                    </a>
                  )}
                  {/* GitHub */}
                  {member.github && (
                    <a
                      href={member.github}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="w-8 h-8 bg-gray-100 dark:bg-gray-800 rounded-full flex items-center justify-center hover:bg-[#1C3D5A] hover:text-white transition-colors"
                    >
                      <svg
                        xmlns="http://www.w3.org/2000/svg"
                        fill="currentColor"
                        viewBox="0 0 24 24"
                        className="w-4 h-4"
                      >
                        <path
                          fillRule="evenodd"
                          d="M12 .5C5.65.5.5 5.65.5 12c0 5.1 3.3 9.4 7.9 10.9.6.1.8-.3.8-.6v-2c-3.2.7-3.9-1.5-3.9-1.5-.6-1.5-1.3-1.9-1.3-1.9-1.1-.8.1-.8.1-.8 1.2.1 1.8 1.2 1.8 1.2 1.1 1.8 2.9 1.3 3.6 1 .1-.8.4-1.3.7-1.6-2.6-.3-5.3-1.3-5.3-5.8 0-1.3.5-2.4 1.2-3.2-.1-.3-.5-1.5.1-3.1 0 0 1-.3 3.2 1.2a11 11 0 0 1 5.8 0c2.2-1.5 3.2-1.2 3.2-1.2.6 1.6.2 2.8.1 3.1.8.8 1.2 1.9 1.2 3.2 0 4.6-2.7 5.5-5.3 5.8.4.4.8 1.1.8 2.2v3.3c0 .3.2.7.8.6a10.9 10.9 0 0 0 7.9-10.9C23.5 5.65 18.35.5 12 .5Z"
                        />
                      </svg>
                    </a>
                  )}
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Timeline Section */}
      <section className="py-20 bg-white dark:bg-[#1E1E1E]">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
            className="text-center mb-16"
          >
            <h2 className="text-3xl md:text-4xl font-bold text-gray-900 dark:text-[#EAEAEA] mb-4">
              Our <span className="gradient-text">Journey</span>
            </h2>
            <p className="text-lg text-gray-600 dark:text-[#A0A0A0]">
              Key milestones in turning our idea into a fully functional product
            </p>
          </motion.div>

          <div className="relative">
            <div className="absolute left-1/2 transform -translate-x-0.5 w-1 h-full bg-gradient-to-b from-[#1C3D5A] to-[#14B8A6]" />
            <div className="space-y-12">
              {milestones.map((milestone, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, x: index % 2 === 0 ? -30 : 30 }}
                  whileInView={{ opacity: 1, x: 0 }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.6, delay: index * 0.1 }}
                  className={`relative flex items-center ${
                    index % 2 === 0 ? "justify-start" : "justify-end"
                  }`}
                >
                  <div className="absolute left-1/2 transform -translate-x-1/2 w-4 h-4 bg-gradient-to-r from-[#1C3D5A] to-[#14B8A6] rounded-full z-10" />
                  <div
                    className={`w-5/12 ${
                      index % 2 === 0 ? "text-right pr-8" : "text-left pl-8"
                    }`}
                  >
                    <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-6">
                      <div className="text-2xl font-bold gradient-text mb-2">
                        {milestone.title}
                      </div>
                      <p className="text-gray-600 dark:text-[#A0A0A0]">
                        {milestone.content}
                      </p>
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
//about.jsx