import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card";
import { Badge } from "../components/ui/badge";
import { Button } from "../components/ui/button";
import { Input } from "../components/ui/input";
import {
  Search,
  MapPin,
  Briefcase,
  Star,
  Mail,
  Phone,
  X,
  ArrowLeft,
  Filter,
  Award,
  Clock
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import lawyersData from "../data/lawyers.json";

export default function LawyerDirectory() {
  const navigate = useNavigate();
  const [lawyers, setLawyers] = useState([]);
  const [filteredLawyers, setFilteredLawyers] = useState([]);
  const [searchTerm, setSearchTerm] = useState("");
  const [selectedCity, setSelectedCity] = useState("All");
  const [selectedSpecialization, setSelectedSpecialization] = useState("All");
  const [minExperience, setMinExperience] = useState(0);
  const [selectedLawyer, setSelectedLawyer] = useState(null);
  const [showContactModal, setShowContactModal] = useState(false);

  // Load lawyers data
  useEffect(() => {
    setLawyers(lawyersData);
    setFilteredLawyers(lawyersData);
  }, []);

  // Get unique cities and specializations for filters
  const cities = ["All", ...new Set(lawyers.map(l => l.city))].sort();
  const specializations = ["All", ...new Set(lawyers.map(l => l.specialization))].sort();

  // Filter lawyers based on search and filters
  useEffect(() => {
    let filtered = lawyers;

    // Search filter
    if (searchTerm) {
      filtered = filtered.filter(lawyer =>
        lawyer.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        lawyer.specialization.toLowerCase().includes(searchTerm.toLowerCase()) ||
        lawyer.city.toLowerCase().includes(searchTerm.toLowerCase())
      );
    }

    // City filter
    if (selectedCity !== "All") {
      filtered = filtered.filter(lawyer => lawyer.city === selectedCity);
    }

    // Specialization filter
    if (selectedSpecialization !== "All") {
      filtered = filtered.filter(lawyer => lawyer.specialization === selectedSpecialization);
    }

    // Experience filter
    filtered = filtered.filter(lawyer => lawyer.experience >= minExperience);

    setFilteredLawyers(filtered);
  }, [searchTerm, selectedCity, selectedSpecialization, minExperience, lawyers]);

  const handleContactLawyer = (lawyer) => {
    setSelectedLawyer(lawyer);
    setShowContactModal(true);
  };

  const handleCloseModal = () => {
    setShowContactModal(false);
    setSelectedLawyer(null);
  };

  const getSpecializationColor = (specialization) => {
    const colors = {
      "Corporate Law": "bg-blue-500/20 text-blue-400 border-blue-500/50",
      "Family Law": "bg-pink-500/20 text-pink-400 border-pink-500/50",
      "Criminal Law": "bg-red-500/20 text-red-400 border-red-500/50",
      "Intellectual Property": "bg-purple-500/20 text-purple-400 border-purple-500/50",
      "Real Estate Law": "bg-green-500/20 text-green-400 border-green-500/50",
      "Labor Law": "bg-yellow-500/20 text-yellow-400 border-yellow-500/50",
      "Tax Law": "bg-orange-500/20 text-orange-400 border-orange-500/50",
      "Consumer Protection": "bg-teal-500/20 text-teal-400 border-teal-500/50",
      "Banking Law": "bg-indigo-500/20 text-indigo-400 border-indigo-500/50",
      "Environmental Law": "bg-emerald-500/20 text-emerald-400 border-emerald-500/50",
    };
    return colors[specialization] || "bg-gray-500/20 text-gray-400 border-gray-500/50";
  };

  return (
    <div className="min-h-screen relative bg-gradient-to-b from-black via-[#0F2A40] to-[#064E3B] overflow-hidden py-16">
      {/* Aurora Background */}
      <div className="absolute inset-0 aurora-bg opacity-20" />

      <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
            <div>
              <h1 className="text-4xl md:text-5xl font-bold text-white mb-2">
                Expert Legal Directory
              </h1>
              <p className="text-lg text-gray-300">
                Connect with verified lawyers across India for your legal needs
              </p>
            </div>
            <Button
              onClick={() => navigate(-1)}
              variant="outline"
              className="border-cyan-400/50 text-cyan-400 hover:bg-cyan-400/10"
            >
              <ArrowLeft className="w-4 h-4 mr-2" />
              Back to Analysis
            </Button>
          </div>
        </motion.div>

        {/* Stats Overview */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="mb-8 grid grid-cols-2 md:grid-cols-4 gap-4"
        >
          <Card className="border-none bg-gradient-to-br from-cyan-900/40 to-blue-900/40 backdrop-blur-md shadow-xl">
            <CardContent className="p-4 text-center">
              <Briefcase className="w-8 h-8 text-cyan-400 mx-auto mb-2" />
              <div className="text-2xl font-bold text-white">{lawyers.length}</div>
              <div className="text-xs text-gray-300">Total Lawyers</div>
            </CardContent>
          </Card>

          <Card className="border-none bg-gradient-to-br from-purple-900/40 to-pink-900/40 backdrop-blur-md shadow-xl">
            <CardContent className="p-4 text-center">
              <MapPin className="w-8 h-8 text-purple-400 mx-auto mb-2" />
              <div className="text-2xl font-bold text-white">{cities.length - 1}</div>
              <div className="text-xs text-gray-300">Cities Covered</div>
            </CardContent>
          </Card>

          <Card className="border-none bg-gradient-to-br from-green-900/40 to-emerald-900/40 backdrop-blur-md shadow-xl">
            <CardContent className="p-4 text-center">
              <Award className="w-8 h-8 text-green-400 mx-auto mb-2" />
              <div className="text-2xl font-bold text-white">{specializations.length - 1}</div>
              <div className="text-xs text-gray-300">Specializations</div>
            </CardContent>
          </Card>

          <Card className="border-none bg-gradient-to-br from-yellow-900/40 to-orange-900/40 backdrop-blur-md shadow-xl">
            <CardContent className="p-4 text-center">
              <Star className="w-8 h-8 text-yellow-400 mx-auto mb-2" />
              <div className="text-2xl font-bold text-white">4.6</div>
              <div className="text-xs text-gray-300">Avg Rating</div>
            </CardContent>
          </Card>
        </motion.div>

        {/* Search and Filters */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="mb-8"
        >
          <Card className="border-none bg-[#064E3B]/90 backdrop-blur-md shadow-xl">
            <CardContent className="p-6">
              <div className="flex flex-col gap-4">
                {/* Search Bar */}
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
                  <Input
                    type="text"
                    placeholder="Search by name, city, or specialization..."
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    className="w-full pl-10 pr-4 py-3 bg-gray-800/50 border border-gray-700 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-cyan-500"
                  />
                </div>

                {/* Filters Row */}
                <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      <MapPin className="w-4 h-4 inline mr-1" />
                      City
                    </label>
                    <select
                      value={selectedCity}
                      onChange={(e) => setSelectedCity(e.target.value)}
                      className="w-full px-4 py-2 bg-gray-800/50 border border-gray-700 rounded-lg text-white focus:outline-none focus:border-cyan-500"
                    >
                      {cities.map((city) => (
                        <option key={city} value={city}>
                          {city}
                        </option>
                      ))}
                    </select>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      <Briefcase className="w-4 h-4 inline mr-1" />
                      Specialization
                    </label>
                    <select
                      value={selectedSpecialization}
                      onChange={(e) => setSelectedSpecialization(e.target.value)}
                      className="w-full px-4 py-2 bg-gray-800/50 border border-gray-700 rounded-lg text-white focus:outline-none focus:border-cyan-500"
                    >
                      {specializations.map((spec) => (
                        <option key={spec} value={spec}>
                          {spec}
                        </option>
                      ))}
                    </select>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      <Clock className="w-4 h-4 inline mr-1" />
                      Min. Experience
                    </label>
                    <select
                      value={minExperience}
                      onChange={(e) => setMinExperience(Number(e.target.value))}
                      className="w-full px-4 py-2 bg-gray-800/50 border border-gray-700 rounded-lg text-white focus:outline-none focus:border-cyan-500"
                    >
                      <option value={0}>Any</option>
                      <option value={5}>5+ Years</option>
                      <option value={10}>10+ Years</option>
                      <option value={15}>15+ Years</option>
                      <option value={20}>20+ Years</option>
                    </select>
                  </div>

                  <div className="flex items-end">
                    <Button
                      onClick={() => {
                        setSearchTerm("");
                        setSelectedCity("All");
                        setSelectedSpecialization("All");
                        setMinExperience(0);
                      }}
                      variant="outline"
                      className="w-full border-gray-600 text-gray-300 hover:bg-gray-800"
                    >
                      <Filter className="w-4 h-4 mr-2" />
                      Reset Filters
                    </Button>
                  </div>
                </div>

                {/* Results Count */}
                <div className="text-center text-gray-300 text-sm">
                  Showing <span className="font-bold text-cyan-400">{filteredLawyers.length}</span> of{" "}
                  <span className="font-bold text-white">{lawyers.length}</span> lawyers
                </div>
              </div>
            </CardContent>
          </Card>
        </motion.div>

        {/* Lawyers Grid */}
        {filteredLawyers.length === 0 ? (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
          >
            <Card className="border-none bg-[#064E3B]/90 backdrop-blur-md shadow-xl">
              <CardContent className="p-12 text-center">
                <Briefcase className="w-16 h-16 text-gray-400 mx-auto mb-4" />
                <h3 className="text-xl font-semibold text-white mb-2">
                  No lawyers found
                </h3>
                <p className="text-gray-300 mb-6">
                  Try adjusting your filters or search terms
                </p>
                <Button
                  onClick={() => {
                    setSearchTerm("");
                    setSelectedCity("All");
                    setSelectedSpecialization("All");
                    setMinExperience(0);
                  }}
                  className="bg-cyan-600 hover:bg-cyan-500 text-white"
                >
                  Reset All Filters
                </Button>
              </CardContent>
            </Card>
          </motion.div>
        ) : (
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            <AnimatePresence>
              {filteredLawyers.map((lawyer, index) => (
                <motion.div
                  key={lawyer.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, scale: 0.9 }}
                  transition={{ delay: 0.3 + index * 0.03 }}
                >
                  <Card className="border-none bg-[#064E3B]/90 backdrop-blur-md shadow-xl hover:shadow-2xl transition-all group h-full flex flex-col">
                    <CardHeader className="border-b border-gray-700/50 pb-4">
                      <div className="flex items-start justify-between gap-3">
                        <div className="flex-1 min-w-0">
                          <CardTitle className="text-white text-lg mb-2 truncate">
                            {lawyer.name}
                          </CardTitle>
                          <div className="flex items-center gap-2 text-gray-400 text-xs mb-2">
                            <MapPin className="w-3 h-3" />
                            <span>{lawyer.city}</span>
                          </div>
                          <div className="flex items-center gap-2">
                            <Star className="w-4 h-4 fill-yellow-400 text-yellow-400" />
                            <span className="text-white font-semibold text-sm">
                              {lawyer.rating}
                            </span>
                            <span className="text-gray-400 text-xs">
                              ({lawyer.experience}+ yrs exp)
                            </span>
                          </div>
                        </div>
                      </div>
                    </CardHeader>

                    <CardContent className="p-4 flex-1 flex flex-col">
                      {/* Specialization Badge */}
                      <div className="mb-4">
                        <Badge className={`border ${getSpecializationColor(lawyer.specialization)}`}>
                          {lawyer.specialization}
                        </Badge>
                      </div>

                      {/* Description */}
                      <p className="text-gray-300 text-sm line-clamp-3 mb-4 flex-1">
                        {lawyer.description}
                      </p>

                      {/* Contact Button */}
                      <Button
                        onClick={() => handleContactLawyer(lawyer)}
                        className="w-full bg-gradient-to-r from-cyan-600 to-cyan-500 hover:from-cyan-500 hover:to-cyan-400 text-white"
                      >
                        <Mail className="w-4 h-4 mr-2" />
                        Contact Lawyer
                      </Button>
                    </CardContent>
                  </Card>
                </motion.div>
              ))}
            </AnimatePresence>
          </div>
        )}
      </div>

      {/* Contact Modal */}
      {showContactModal && selectedLawyer && (
        <div className="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center z-50 p-4">
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className="bg-gradient-to-br from-[#064E3B] to-[#0F2A40] rounded-xl shadow-2xl max-w-2xl w-full border border-cyan-400/30"
          >
            <div className="p-6">
              {/* Modal Header */}
              <div className="flex items-center justify-between mb-6">
                <h3 className="text-2xl font-bold text-white flex items-center gap-2">
                  <Briefcase className="w-6 h-6 text-cyan-400" />
                  Lawyer Details
                </h3>
                <button
                  onClick={handleCloseModal}
                  className="text-gray-400 hover:text-white transition-colors"
                >
                  <X className="w-6 h-6" />
                </button>
              </div>

              {/* Lawyer Info */}
              <div className="space-y-6">
                {/* Name and Rating */}
                <div>
                  <h4 className="text-2xl font-bold text-white mb-2">
                    {selectedLawyer.name}
                  </h4>
                  <div className="flex items-center gap-4 flex-wrap">
                    <div className="flex items-center gap-2">
                      <Star className="w-5 h-5 fill-yellow-400 text-yellow-400" />
                      <span className="text-white font-semibold">
                        {selectedLawyer.rating}/5.0
                      </span>
                    </div>
                    <Badge className={`border ${getSpecializationColor(selectedLawyer.specialization)}`}>
                      {selectedLawyer.specialization}
                    </Badge>
                  </div>
                </div>

                {/* Location and Experience */}
                <div className="grid md:grid-cols-2 gap-4">
                  <div className="bg-gray-800/50 p-4 rounded-lg border border-gray-700">
                    <div className="flex items-center gap-3 mb-2">
                      <MapPin className="w-5 h-5 text-cyan-400" />
                      <span className="text-gray-400 text-sm">Location</span>
                    </div>
                    <p className="text-white font-semibold">{selectedLawyer.city}</p>
                  </div>

                  <div className="bg-gray-800/50 p-4 rounded-lg border border-gray-700">
                    <div className="flex items-center gap-3 mb-2">
                      <Clock className="w-5 h-5 text-cyan-400" />
                      <span className="text-gray-400 text-sm">Experience</span>
                    </div>
                    <p className="text-white font-semibold">{selectedLawyer.experience}+ Years</p>
                  </div>
                </div>

                {/* Description */}
                <div className="bg-gray-800/30 p-4 rounded-lg border border-gray-700">
                  <h5 className="text-white font-semibold mb-2 flex items-center gap-2">
                    <Award className="w-5 h-5 text-purple-400" />
                    About
                  </h5>
                  <p className="text-gray-300 text-sm leading-relaxed">
                    {selectedLawyer.description}
                  </p>
                </div>

                {/* Contact Information */}
                <div className="bg-gradient-to-r from-cyan-900/40 to-blue-900/40 p-4 rounded-lg border border-cyan-500/30">
                  <h5 className="text-white font-semibold mb-3">Contact Information</h5>
                  <div className="space-y-3">
                    <div className="flex items-center gap-3">
                      <Phone className="w-5 h-5 text-cyan-400" />
                      <a
                        href={`tel:${selectedLawyer.contact}`}
                        className="text-gray-200 hover:text-cyan-400 transition-colors"
                      >
                        {selectedLawyer.contact}
                      </a>
                    </div>
                    <div className="flex items-center gap-3">
                      <Mail className="w-5 h-5 text-cyan-400" />
                      <a
                        href={`mailto:${selectedLawyer.email}`}
                        className="text-gray-200 hover:text-cyan-400 transition-colors"
                      >
                        {selectedLawyer.email}
                      </a>
                    </div>
                  </div>
                </div>

                {/* Action Buttons */}
                <div className="flex gap-3 pt-2">
                  <Button
                    onClick={() => window.open(`tel:${selectedLawyer.contact}`)}
                    className="flex-1 bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-500 hover:to-emerald-500 text-white"
                  >
                    <Phone className="w-4 h-4 mr-2" />
                    Call Now
                  </Button>
                  <Button
                    onClick={() => window.open(`mailto:${selectedLawyer.email}`)}
                    className="flex-1 bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white"
                  >
                    <Mail className="w-4 h-4 mr-2" />
                    Send Email
                  </Button>
                  <Button
                    onClick={handleCloseModal}
                    variant="outline"
                    className="border-gray-600 text-gray-300 hover:bg-gray-800"
                  >
                    Close
                  </Button>
                </div>
              </div>
            </div>
          </motion.div>
        </div>
      )}
    </div>
  );
}